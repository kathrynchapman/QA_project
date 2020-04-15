from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
from argparse import ArgumentParser
import torch
from transformers import BertPreTrainedModel, BertConfig, BertTokenizer
from bert_model import BertModel
import torch.nn as nn
from torch.autograd import Variable


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def bert_rep(args, bert_config, is_training, input_ids, input_mask, segment_ids, history_answer_marker, use_one_hot_embeddings):
    """
    :param bert_config: 'BertConfig' instance
    :param is_training: bool; not sure if necessary?
    :param input_ids: torch.LongTensor of shape (batch_size, sequence_length),
    indices of input sequence tokens in the vocabulary, which can be obtained using transformers.BertTokenizer
    :param input_mask: aka attention mask; The attention mask is an optional argument used when
    batching sequences together. This argument indicates to the model which tokens should be attended to,
    and which should not.
    :param segment_ids: aka token type ids; indicates whether a token is part of a context or the question
    :param history_answer_marker: see FeatureDescriptions.pdf
    :param use_one_hot_embeddings:
    :return sequence_output: torch.FloatTensor of shape (batch_size, sequence_length, hidden_size),
    sequence of hidden-states at the output of the last layer of the model
    :return pooled_output: torch.FloatTensor of shape (batch_size, hidden_size),
    last layer hidden-state of the first token of the sequence (classification token) further processed by
    a Linear layer and a Tanh activation function
    """
    #
    # model = BertModel(bert_config, args).to('cuda:1')
    #
    # inputs = {
    #     # "config":bert_config,
    #     "input_ids": input_ids.to('cuda:1'),
    #     "attention_mask": input_mask.to('cuda:1'),
    #     "token_type_ids": segment_ids.to('cuda:1'),
    #     "history_answer_marker": history_answer_marker.to('cuda:1'),
    # }
    #
    # outputs = model(**inputs)



    model = BertModel(bert_config, args)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)

    inputs = {
        # "config":bert_config,
        "input_ids": input_ids.to(args.device),
        "attention_mask": input_mask.to(args.device),
        "token_type_ids": segment_ids.to(args.device),
        "history_answer_marker": history_answer_marker.to(args.device),
    }

    outputs = model(**inputs)
    sequence_output = outputs[0]  # final hidden layer, with dimensions [batch_size, max_seq_len, hidden_size]
    pooled_output = outputs[1]  # entire sequence representation/embedding of 'CLS' token
    # print("CLS:", final_hidden.shape)
    # print(print("sent_rep:", sent_rep.shape)
    return sequence_output, pooled_output


class CQAModel(nn.Module):
    def __init__(self, args):
        super(CQAModel, self).__init__()
        self.args = args
        self.output_weights = nn.Parameter(torch.empty(2, self.args.bert_hidden).normal_(mean=0, std=0.02))
        # self.output_weights = nn.Parameter(truncated_normal_(torch.empty(2, self.args.bert_hidden)))
        self.output_bias = nn.Parameter(torch.zeros(2))

    def forward(self, final_hidden):
        final_hidden_shape = final_hidden.shape
        batch_size = final_hidden_shape[0]
        seq_length = final_hidden_shape[1]
        hidden_size = final_hidden_shape[2]

        final_hidden_matrix = final_hidden.view(batch_size * seq_length, hidden_size)


        logits = torch.matmul(final_hidden_matrix, self.output_weights.T)

        logits = torch.add(logits, self.output_bias)

        logits = logits.reshape(batch_size, seq_length, 2)
        logits = logits.permute(2, 0, 1)
        # logits = logits.T

        unstacked_logits = torch.unbind(logits, dim=0)

        (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

        return start_logits, end_logits


class YesNoModel(nn.Module):
    def __init__(self, args):
        super(YesNoModel, self).__init__()
        self.args = args
        self.linear_layer = torch.nn.Linear(self.args.bert_hidden, 3, bias=False)
        torch.nn.init.normal_(self.linear_layer.weight, std=0.02)

    def forward(self, sent_rep):
        logits = self.linear_layer(sent_rep)
        return logits

class FollowUpModel(nn.Module):
    def __init__(self, args):
        super(FollowUpModel, self).__init__()
        self.args = args
        self.linear_layer = torch.nn.Linear(self.args.bert_hidden, 3, bias=False)
        torch.nn.init.normal_(self.linear_layer.weight, std=0.02)

    def forward(self, sent_rep):
        logits = self.linear_layer(sent_rep)
        return logits


class HistoryAttentionNet(torch.nn.Module):
    """
    :param bert_representation: torch.Tensor of shape (batch_size, max_seq_length, hidden_size),
        token-level representation
    :param history_attention_input: torch.Tensor of shape (batch_size, hidden_size),
        sequence-level representation, obtained by averaging the token-level representation along axis 1 (max_seq_length)
    :param mtl_input: torch.Tensor of shape (batch_size, hidden_size),
        sequence-level representation, obtained by averaging the token-level representation along axis 1 (max_seq_length),
        same as history_attention_input
    :param slice_mask: list containing integers that indicate the size of each subtensor we will get after splitting
        the history_attention_input tensor, corresponding to different examples/subpassages/padding
    :param slice_num: int representing the number of examples/sub-passages in the batch
    :return new_bert_representation: torch.Tensor of shape (batch_size, max_seq_length, hidden_size),
        aggregated token-level representation
    :return new_mtl_input: torch.Tensor of shape (batch_size, hidden_size), aggregated sequence-level representation
    :return probs: torch.Tensor of shape (batch_size, max_considered_history_turns, 1), containing the attention weights
        of each variation to the aggregated representation of its example/sub-passage
    """

    # Example with the following arguments:
    # batch_size = 8
    # max_seq_length = 12
    # hidden_size = 4
    # slice_mask: [3, 3, 2, 1, 1, 1, 1, 1]
    # slice_num = 3

    #### GENERATE TENSOR: probs ####

    def __init__(self, args):
        super(HistoryAttentionNet, self).__init__()
        self.args = args
        # create network layers
        # layer_linear = torch.nn.Linear(input_tensor.shape[2], 1).to(args.device)
        self.layer_linear = torch.nn.Linear(self.args.bert_hidden, 1)
        torch.nn.init.normal_(self.layer_linear.weight,
                              std=0.02)  # initialize the weights to a normal distribution (the original TensorFlow code uses a truncated normal distribution)
        # self.layer_linear = nn.DataParallel(self.layer_linear)


    def pad_fn(self, x, num):
        # pad the splits so that all of them are of size (11, 4)
        padding = torch.nn.ZeroPad2d((0, 0, self.args.max_considered_history_turns - num, 0))  # padding at the top
        return padding(x)

    def sequence_mask(self, lengths, maxlen):
        """
        Returns a mask tensor representing the first n positions of each cell.
        Equivalent to tf.sequence_mask() with param dtype=tf.float32.
        """
        if maxlen is None:
            maxlen = lengths.max()
        mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
        mask = mask.numpy().astype('float32')
        mask = torch.from_numpy(mask)
        return mask

    def flip(self, x, dim):
        """
        Reverses specific dimensions of a tensor.
        Equivalent to tf.reverse().
        """
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def forward(self, bert_representation, history_attention_input, mtl_input, slice_mask, slice_num):
        # pad history_attention_input: (8, 3) --> (13, 3)
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.args.batch_size - slice_num)) # padding at the bottom
        history_attention_input = padding(history_attention_input)

        # split history_attention_input into 8 tensors of sizes: (3, 4), (3, 4), (2, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)
        splits = torch.split(history_attention_input, slice_mask, 0)

        padded = []
        for i in range(self.args.batch_size):
            padded.append(self.pad_fn(splits[i], slice_mask[i]))

        # stack the splits to form an input_tensor of size (8, 11, 4)
        input_tensor = torch.stack(padded, axis=0)

        # pass input_tensor to a single-layer feed-forward neural network, after which the input_tensor will be of size (8, 11, 1)


        # do the forward pass
        # print("Input tesnor device:", input_tensor.get_device())
        # print("Linear layer device:", self.layer_linear)
        logits = self.layer_linear(input_tensor)

        # squeeze input_tensor along dimension 2, so that it has size (8, 11)
        logits = torch.squeeze(logits, dim=2)

        # mask the padded parts of input_tensor out and apply the exponential function to all its cells



        logits_mask = self.sequence_mask(torch.tensor(slice_mask), self.args.max_considered_history_turns)
        logits_mask = self.flip(logits_mask, 1)
        # exp_logits_masked = torch.exp(logits) * logits_mask.to(args.device)
        exp_logits_masked = torch.exp(logits) * logits_mask.to('cuda:0')
        # use the softmax function to generate a tensor of probabilities
        # slice the resulting exp_logits_masked tensor to get rid of the padded rows, obtaining a tensor of size (3, 11)
        exp_logits_masked = exp_logits_masked[:slice_num, :]
        # divide each cell by the sum of all cells
        probs = exp_logits_masked / torch.sum(exp_logits_masked, dim=1, keepdim=True)

        # unsqueeze the probs tensor so that it has size (3, 11, 1)
        probs = torch.unsqueeze(probs, dim=-1)

        #### GENERATE TENSOR: new_mtl_input ####

        # pad mtl_input: (8, 3) --> (13, 3)
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.args.batch_size - slice_num))  # padding at the bottom
        mtl_input = padding(mtl_input)

        # split mtl_input into tensors of sizes: (3, 4), (3, 4), (2, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)
        splits = torch.split(mtl_input, slice_mask, 0)

        # pad the splits so that all of them are of size (11, 4)
        padded = []
        for i in range(self.args.batch_size):
            padded.append(self.pad_fn(splits[i], slice_mask[i]))

        # stack the splits to form a tensor of size (8, 11, 4)
        mtl_input = torch.stack(padded, axis=0)

        # slice mtl_input to get rid of the paddings, resulting in a tensor of size (3, 11, 4)
        mtl_input = mtl_input[:slice_num, :, :]

        # multiply by probs and sum along dimension 1, resulting in a new_mtl_input tensor of size (3, 4)
        new_mtl_input = torch.sum(mtl_input * probs, dim=1)

        #### GENERATE TENSOR: new_bert_representation ####

        # pad bert_representation: (8, 12, 4) --> (13, 12, 4)
        bert_representation = torch.nn.functional.pad(bert_representation,
                                                  (0, 0, 0, 0, 0, self.args.batch_size - slice_num))  # padding at the back

        # split mtl_input into tensors of sizes: (3, 12, 4), (3, 12, 4), (2, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4)
        splits = torch.split(bert_representation, slice_mask, 0)

        # pad the splits so that all of them are of size (11, 12, 4)
        pad_fn = lambda x, num: torch.nn.functional.pad(x, (
        0, 0, 0, 0, self.args.max_considered_history_turns - num, 0))  # padding at the front
        padded = []
        for i in range(self.args.batch_size):
            padded.append(pad_fn(splits[i], slice_mask[i]))

        # stack the splits to form a token_tensor of size (8, 11, 12, 4)
        token_tensor = torch.stack(padded, axis=0)

        # slice token tensor to get rid of the padded part, resulting in a tensor of size (3, 11, 12, 4)
        token_tensor = token_tensor[:slice_num, :, :, :]

        # unsqueeze the probs tensor: (3, 11, 1) --> (3, 11, 1, 1)
        probs = torch.unsqueeze(probs, dim=-1)

        # multiply token_tensor by probs and sum along dimension 1
        new_bert_representation = torch.sum(token_tensor * probs, dim=1)

        # squeeze back the probs tensor: (3, 11, 1, 1) --> (3, 11, 1)
        probs = torch.squeeze(probs)

        return new_bert_representation, new_mtl_input, probs



#------------------------------------------------------------------------------------------------------------

#
#
# class MTLModel(torch.nn.Module):
#     def __init__(self, args):
#         super(MTLModel, self).__init__()
#         self.args = args
#
#         self.cqa = CQAModel(self.args).to(args.device)
#         # self.cqa = nn.DataParallel(self.cqa)
#         self.yesno = YesNoModel(self.args).to(args.device)
#         # self.yesno = nn.DataParallel(self.yesno)
#         self.followup = FollowUpModel(self.args).to(args.device)
#         # self.followup = nn.DataParallel(self.followup)
#
#     def forward(self, final_hidden, sentence_rep):
#         return (self.cqa(final_hidden), self.yesno(sentence_rep), self.followup(sentence_rep),)



class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cuda:1'))



class MTLModel(nn.Module):
    def __init__(self, args):
        super(MTLModel, self).__init__()
        self.args = args
        self.cqa = CQAModel(self.args).to(args.device)
        self.yesno = YesNoModel(self.args).to(args.device)
        self.followup = FollowUpModel(self.args).to(args.device)
        self.ham = HistoryAttentionNet(self.args).to(args.device)


        # self.ham = HistoryAttentionNet(self.args).to('cuda:0')
        # self.cqa = CQAModel(self.args).to('cuda:2')
        # self.yesno = YesNoModel(self.args).to('cuda:3')
        # self.followup = FollowUpModel(self.args).to('cuda:3')

    def forward(self, bert_representation, history_attention_input, mtl_input, batch_slice_mask, batch_slice_num):


        new_bert_representation, new_mtl_input, attention_weights = self.ham(bert_representation.to(self.args.device),
                                                                                    history_attention_input.to(self.args.device),
                                                                                    mtl_input.to(self.args.device),
                                                                                    batch_slice_mask,
                                                                                    batch_slice_num)


        return (self.cqa(new_bert_representation), self.yesno(new_mtl_input), self.followup(new_mtl_input),)




    # def forward(self, bert_representation, history_attention_input, mtl_input, batch_slice_mask, batch_slice_num):
    #
    #
    #     new_bert_representation, new_mtl_input, attention_weights = self.ham(bert_representation.to('cuda:0'),
    #                                                                                 history_attention_input.to('cuda:0'),
    #                                                                                 mtl_input.to('cuda:0'),
    #                                                                                 batch_slice_mask,
    #                                                                                 batch_slice_num)
    #
    #
    #     return (self.cqa(new_bert_representation.to('cuda:2')), self.yesno(new_mtl_input.to('cuda:3')), self.followup(new_mtl_input.to('cuda:3')),)
#------------------------------------------------------------------------------------------------------------



# def cqa_model(final_hidden):
#     """
#     :param final_hidden: torch.FloatTensor of shape (batch_size, sequence_length, hidden_size),
#         sequence of hidden-states at the output of the last layer of the model
#     :return start_logits: torch.Tensor object
#     :return end_logits: torch.Tensor object
#     """
#
#     final_hidden_shape = final_hidden.shape
#     batch_size = final_hidden_shape[0]
#     seq_length = final_hidden_shape[1]
#     hidden_size = final_hidden_shape[2]
#
#     #In the original code, they use truncated normal distribution, but there's no function for this in pytorch
#     #I found this workaround https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
#     #But maybe it's not worth it to go through all that trouble, so I just used normal distribution for now
#     output_weights = torch.empty(2, 768).normal_(std=0.02)
#
#     output_bias = torch.zeros(2)
#
#     final_hidden_matrix = final_hidden.view(batch_size * seq_length, hidden_size)
#     logits = torch.matmul(final_hidden_matrix, torch.transpose(output_weights, 0, 1))
#     logits = torch.add(logits, output_bias)
#
#     logits = logits.reshape(batch_size, seq_length, 2)
#     logits = logits.permute(2, 0, 1)
#     # logits = logits.T
#
#     unstacked_logits = torch.unbind(logits, dim=0)
#
#     (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
#
#     return start_logits, end_logits
#
#
# def yesno_model(sent_rep):
#     """
#     :param sent_rep: torch.FloatTensor of shape (batch_size, hidden_size),
#         last layer hidden-state of the first token of the sequence (classification token) further processed by
#         a Linear layer and a Tanh activation function
#     :return logits: torch.Tensor object
#     """
#
#     linear_layer = torch.nn.Linear(sent_rep.shape[1], 3)
#
#     #Initialize the weights to a normal distribution with sd=0.02 (IN THE ORIGINAL CODE, THEY USE TRUNCATED NORMAL DISTRIBUTION)
#     torch.nn.init.normal_(linear_layer.weight, std=0.02)
#
#     logits = linear_layer(sent_rep)
#
#     return logits
#
#
# def followup_model(sent_rep):
#     """
#     :param sent_rep: torch.FloatTensor of shape (batch_size, hidden_size),
#         last layer hidden-state of the first token of the sequence (classification token) further processed by
#         a Linear layer and a Tanh activation function
#     :return logits: torch.Tensor object
#     """
#
#     linear_layer = torch.nn.Linear(sent_rep.shape[1], 3)
#
#     #Initialize the weights to a normal distribution with sd=0.02 (IN THE ORIGINAL CODE, THEY USE TRUNCATED NORMAL DISTRIBUTION)
#     torch.nn.init.normal_(linear_layer.weight, std=0.02)
#
#     logits = linear_layer(sent_rep)
#
#     return logits

# def history_attention_net(args, bert_representation, history_attention_input, mtl_input, slice_mask, slice_num):
#     """
#     :param bert_representation: torch.Tensor of shape (batch_size, max_seq_length, hidden_size),
#         token-level representation
#     :param history_attention_input: torch.Tensor of shape (batch_size, hidden_size),
#         sequence-level representation, obtained by averaging the token-level representation along axis 1 (max_seq_length)
#     :param mtl_input: torch.Tensor of shape (batch_size, hidden_size),
#         sequence-level representation, obtained by averaging the token-level representation along axis 1 (max_seq_length),
#         same as history_attention_input
#     :param slice_mask: list containing integers that indicate the size of each subtensor we will get after splitting
#         the history_attention_input tensor, corresponding to different examples/subpassages/padding
#     :param slice_num: int representing the number of examples/sub-passages in the batch
#     :return new_bert_representation: torch.Tensor of shape (batch_size, max_seq_length, hidden_size),
#         aggregated token-level representation
#     :return new_mtl_input: torch.Tensor of shape (batch_size, hidden_size), aggregated sequence-level representation
#     :return probs: torch.Tensor of shape (batch_size, max_considered_history_turns, 1), containing the attention weights
#         of each variation to the aggregated representation of its example/sub-passage
#     """
#
#     # Example with the following arguments:
#     # batch_size = 8
#     # max_seq_length = 12
#     # hidden_size = 4
#     # slice_mask: [3, 3, 2, 1, 1, 1, 1, 1]
#     # slice_num = 3
#
#
#     #### GENERATE TENSOR: probs ####
#
#     # pad history_attention_input: (8, 3) --> (13, 3)
#     padding = torch.nn.ZeroPad2d((0, 0, 0, args.batch_size-slice_num)) # padding at the bottom
#     history_attention_input = padding(history_attention_input)
#
#     # split history_attention_input into 8 tensors of sizes: (3, 4), (3, 4), (2, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)
#     splits = torch.split(history_attention_input, slice_mask, 0)
#
#     # pad the splits so that all of them are of size (11, 4)
#     def pad_fn(x, num):
#         padding = torch.nn.ZeroPad2d((0, 0, args.max_considered_history_turns-num, 0)) # padding at the top
#         return padding(x)
#
#     padded = []
#     for i in range(args.batch_size):
#         padded.append(pad_fn(splits[i], slice_mask[i]))
#
#     # stack the splits to form an input_tensor of size (8, 11, 4)
#     input_tensor = torch.stack(padded, axis=0)
#
#
#     # pass input_tensor to a single-layer feed-forward neural network, after which the input_tensor will be of size (8, 11, 1)
#
#     # create network layers
#     # layer_linear = torch.nn.Linear(input_tensor.shape[2], 1).to(args.device)
#     layer_linear = torch.nn.Linear(input_tensor.shape[2], 1)
#     torch.nn.init.normal_(layer_linear.weight, std=0.02) # initialize the weights to a normal distribution (the original TensorFlow code uses a truncated normal distribution)
#     # do the forward pass
#     logits = layer_linear(input_tensor)
#
#     # squeeze input_tensor along dimension 2, so that it has size (8, 11)
#     logits = torch.squeeze(logits, dim=2)
#
#     # mask the padded parts of input_tensor out and apply the exponential function to all its cells
#
#     def sequence_mask(lengths, maxlen):
#         """
#         Returns a mask tensor representing the first n positions of each cell.
#         Equivalent to tf.sequence_mask() with param dtype=tf.float32.
#         """
#         if maxlen is None:
#             maxlen = lengths.max()
#         mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
#         mask = mask.numpy().astype('float32')
#         mask = torch.from_numpy(mask)
#         return mask
#
#     def flip(x, dim):
#         """
#         Reverses specific dimensions of a tensor.
#         Equivalent to tf.reverse().
#         """
#         indices = [slice(None)] * x.dim()
#         indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
#                                     dtype=torch.long, device=x.device)
#         return x[tuple(indices)]
#
#     logits_mask = sequence_mask(torch.tensor(slice_mask), args.max_considered_history_turns)
#     logits_mask = flip(logits_mask, 1)
#     # exp_logits_masked = torch.exp(logits) * logits_mask.to(args.device)
#     exp_logits_masked = torch.exp(logits) * logits_mask
#     # use the softmax function to generate a tensor of probabilities
#     # slice the resulting exp_logits_masked tensor to get rid of the padded rows, obtaining a tensor of size (3, 11)
#     exp_logits_masked = exp_logits_masked[:slice_num, :]
#     # divide each cell by the sum of all cells
#     probs = exp_logits_masked / torch.sum(exp_logits_masked, dim=1, keepdim=True)
#
#     # unsqueeze the probs tensor so that it has size (3, 11, 1)
#     probs = torch.unsqueeze(probs, dim=-1)
#
#
#     #### GENERATE TENSOR: new_mtl_input ####
#
#     # pad mtl_input: (8, 3) --> (13, 3)
#     padding = torch.nn.ZeroPad2d((0, 0, 0, args.batch_size-slice_num)) #padding at the bottom
#     mtl_input = padding(mtl_input)
#
#     # split mtl_input into tensors of sizes: (3, 4), (3, 4), (2, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)
#     splits = torch.split(mtl_input, slice_mask, 0)
#
#     # pad the splits so that all of them are of size (11, 4)
#     padded = []
#     for i in range(args.batch_size):
#         padded.append(pad_fn(splits[i], slice_mask[i]))
#
#     # stack the splits to form a tensor of size (8, 11, 4)
#     mtl_input = torch.stack(padded, axis=0)
#
#     # slice mtl_input to get rid of the paddings, resulting in a tensor of size (3, 11, 4)
#     mtl_input = mtl_input[:slice_num, :, :]
#
#     # multiply by probs and sum along dimension 1, resulting in a new_mtl_input tensor of size (3, 4)
#     new_mtl_input = torch.sum(mtl_input * probs, dim=1)
#
#
#     #### GENERATE TENSOR: new_bert_representation ####
#
#     # pad bert_representation: (8, 12, 4) --> (13, 12, 4)
#     bert_representation = torch.nn.functional.pad(bert_representation, (0, 0, 0, 0, 0, args.batch_size-slice_num)) #padding at the back
#
#     # split mtl_input into tensors of sizes: (3, 12, 4), (3, 12, 4), (2, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4)
#     splits = torch.split(bert_representation, slice_mask, 0)
#
#     # pad the splits so that all of them are of size (11, 12, 4)
#     pad_fn = lambda x, num: torch.nn.functional.pad(x, (0, 0, 0, 0, args.max_considered_history_turns-num, 0)) #padding at the front
#     padded = []
#     for i in range(args.batch_size):
#         padded.append(pad_fn(splits[i], slice_mask[i]))
#
#     # stack the splits to form a token_tensor of size (8, 11, 12, 4)
#     token_tensor = torch.stack(padded, axis=0)
#
#     # slice token tensor to get rid of the padded part, resulting in a tensor of size (3, 11, 12, 4)
#     token_tensor = token_tensor[:slice_num, :, :, :]
#
#     # unsqueeze the probs tensor: (3, 11, 1) --> (3, 11, 1, 1)
#     probs = torch.unsqueeze(probs, dim=-1)
#
#     # multiply token_tensor by probs and sum along dimension 1
#     new_bert_representation = torch.sum(token_tensor * probs, dim=1)
#
#     # squeeze back the probs tensor: (3, 11, 1, 1) --> (3, 11, 1)
#     probs = torch.squeeze(probs)
#
#     return new_bert_representation, new_mtl_input, probs


def disable_history_attention_net(args, bert_representation, history_attention_input, mtl_input, slice_mask, slice_num):
    """
    :param bert_representation: torch.Tensor of shape (batch_size, max_seq_length, hidden_size), 
        token-level representation
    :param history_attention_input: torch.Tensor of shape (batch_size, hidden_size), 
        sequence-level representation, obtained by averaging the token-level representation along axis 1 (max_seq_length)
    :param mtl_input: torch.Tensor of shape (batch_size, hidden_size), 
        sequence-level representation, obtained by averaging the token-level representation along axis 1 (max_seq_length),
        same as history_attention_input
    :param slice_mask: list containing integers that indicate the size of each subtensor we will get after splitting
        the history_attention_input tensor, corresponding to different examples/subpassages/padding
    :param slice_num: int representing the number of examples/sub-passages in the batch
    :return new_bert_representation: torch.Tensor of shape (batch_size, max_seq_length, hidden_size),
        aggregated token-level representation
    :return new_mtl_input: torch.Tensor of shape (batch_size, hidden_size), aggregated sequence-level representation
    :return probs: torch.Tensor of shape (batch_size, max_considered_history_turns, 1), containing the attention weights
        of each variation to the aggregated representation of its example/sub-passage
    """

    # Example with the following arguments:
    # batch_size = 8
    # max_seq_length = 12
    # hidden_size = 4
    # slice_mask: [3, 3, 2, 1, 1, 1, 1, 1]
    # slice_num = 3


    #### GENERATE TENSOR: probs ####

    # pad history_attention_input: (8, 3) --> (13, 3)
    padding = torch.nn.ZeroPad2d((0, 0, 0, args.batch_size-slice_num)) #padding at the bottom
    history_attention_input = padding(history_attention_input)
    
    # split history_attention_input into 8 tensors of sizes: (3, 4), (3, 4), (2, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)
    splits = torch.split(history_attention_input, slice_mask, 0)

    # pad the splits so that all of them are of size (11, 4)
    def pad_fn(x, num):
        padding = torch.nn.ZeroPad2d((0, 0, args.max_considered_history_turns-num, 0)) #padding at the top
        return padding(x) 
        
    padded = []
    for i in range(args.batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
    
    # stack the splits to form an input_tensor of size (8, 11, 4)
    input_tensor = torch.stack(padded, axis=0)

    # we assign equal logits, which means equal attention weights
    # this helps us to see whether the attention networks as expected
    logits = torch.ones(args.batch_size, args.max_considered_history_turns)    
    
    # mask the padded parts of input_tensor out and apply the exponential function to all its cells
    def sequence_mask(lengths, maxlen):
        """
        Returns a mask tensor representing the first n positions of each cell.
        Equivalent to tf.sequence_mask() with param dtype=tf.float32.
        """
        if maxlen is None:
            maxlen = lengths.max()
        mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
        mask = mask.numpy().astype('float32')
        mask = torch.from_numpy(mask)
        return mask

    def flip(x, dim):
        """
        Reverses specific dimensions of a tensor.
        Equivalent to tf.reverse().
        """
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    logits_mask = sequence_mask(torch.tensor(slice_mask), args.max_considered_history_turns)
    logits_mask = flip(logits_mask, 1)
    exp_logits_masked = torch.exp(logits) * logits_mask
    
    # use the softmax function to generate a tensor of probabilities  
    # slice the resulting exp_logits_masked tensor to get rid of the padded rows, obtaining a tensor of size (3, 11)
    exp_logits_masked = exp_logits_masked[:slice_num, :]
    # divide each cell by the sum of all cells
    probs = exp_logits_masked / torch.sum(exp_logits_masked, dim=1, keepdim=True)
    
    # e.g. 4 * 11 * 768
    input_tensor = input_tensor[:slice_num, :, :]
    
    # unsqueeze the probs tensor so that it has size (3, 11, 1)
    probs = torch.unsqueeze(probs, dim=-1)


    #### GENERATE TENSOR: new_mtl_input ####

    # pad mtl_input: (8, 3) --> (13, 3)
    padding = torch.nn.ZeroPad2d((0, 0, 0, args.batch_size-slice_num)) #padding at the bottom
    mtl_input = padding(mtl_input)

    # split mtl_input into tensors of sizes: (3, 4), (3, 4), (2, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)    
    splits = torch.split(mtl_input, slice_mask, 0) 

    # pad the splits so that all of them are of size (11, 4)
    padded = []
    for i in range(args.batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))

    # stack the splits to form a tensor of size (8, 11, 4) 
    mtl_input = torch.stack(padded, axis=0)

    # slice mtl_input to get rid of the paddings, resulting in a tensor of size (3, 11, 4)
    mtl_input = mtl_input[:slice_num, :, :]    
    
    # multiply by probs and sum along dimension 1, resulting in a new_mtl_input tensor of size (3, 4)
    new_mtl_input = torch.sum(mtl_input * probs, dim=1)
    
    
    #### GENERATE TENSOR: new_bert_representation ####

    # pad bert_representation: (8, 12, 4) --> (13, 12, 4) 
    bert_representation = torch.nn.functional.pad(bert_representation, (0, 0, 0, 0, 0, args.batch_size-slice_num)) #padding at the back
    
    # split mtl_input into tensors of sizes: (3, 12, 4), (3, 12, 4), (2, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4)        
    splits = torch.split(bert_representation, slice_mask, 0) 

    # pad the splits so that all of them are of size (11, 12, 4)
    pad_fn = lambda x, num: torch.nn.functional.pad(x, (0, 0, 0, 0, args.max_considered_history_turns-num, 0)) #padding at the front
    padded = []
    for i in range(args.batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
        
    # stack the splits to form a token_tensor of size (8, 11, 12, 4)
    token_tensor = torch.stack(padded, axis=0)

    # slice token tensor to get rid of the padded part, resulting in a tensor of size (3, 11, 12, 4)
    token_tensor = token_tensor[:slice_num, :, :, :]
    
    # unsqueeze the probs tensor: (3, 11, 1) --> (3, 11, 1, 1)
    probs = torch.unsqueeze(probs, dim=-1)
    
    # multiply token_tensor by probs and sum along dimension 1
    new_bert_representation = torch.sum(token_tensor * probs, dim=1)

    # squeeze back the probs tensor: (3, 11, 1, 1) --> (3, 11, 1)
    probs = torch.squeeze(probs)
    
    return new_bert_representation, new_mtl_input, probs


def fine_grained_history_attention_net(args, bert_representation, mtl_input, slice_mask, slice_num):
    """
    :param bert_representation: torch.Tensor of shape (batch_size, max_seq_length, hidden_size), 
        token-level representation
    :param mtl_input: torch.Tensor of shape (batch_size, hidden_size), 
        sequence-level representation, obtained by averaging the token-level representation along axis 1 (max_seq_length)
    :param slice_mask: list containing integers that indicate the size of each subtensor we will get after splitting
        the history_attention_input tensor, corresponding to different examples/subpassages/padding
    :param slice_num: int representing the number of examples/sub-passages in the batch
    :return new_bert_representation: torch.Tensor of shape (batch_size, max_seq_length, hidden_size),
        aggregated token-level representation
    :return new_mtl_input: torch.Tensor of shape (batch_size, hidden_size), aggregated sequence-level representation
    :return probs: torch.Tensor of shape (batch_size, max_considered_history_turns, 1), containing the attention weights
        of each variation to the aggregated representation of its example/sub-passage
    """

    # Example with the following arguments:
    # batch_size = 8
    # max_seq_length = 12
    # hidden_size = 4
    # slice_mask: [3, 3, 2, 1, 1, 1, 1, 1]
    # slice_num = 3

    # first concat the bert_representation and mtl_input together
    # so that we can process them together
    # shape for bert_representation: 12 * 384 * 768, shape for mtl_input: 12 * 768
    # after concat: 12 * 385 * 768
    
    # 12 * 385 * 768 --> 20 * 385 * 768
    bert_representation = torch.cat((bert_representation, torch.unsqueeze(mtl_input, dim=1)), dim=1)
    bert_representation = torch.nn.functional.pad(bert_representation, (0, 0, 0, 0, 0, args.batch_size-slice_num)) #padding at the back
    splits = torch.split(bert_representation, slice_mask, 0)

    pad_fn = lambda x, num: torch.nn.functional.pad(x, (0, 0, 0, 0, args.max_considered_history_turns - num, 0)) #padding at the front
    padded = []
    for i in range(args.batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))

    # --> 12 * 11 * 385 * 768
    token_tensor = torch.stack(padded, axis=0)

    # --> 12 * 385 * 11 * 768
    token_tensor_t = token_tensor.permute(0, 2, 1, 3)

    # --> 12 * 385 * 11 * 1
    #Create network layers
    layer_linear = torch.nn.Linear(token_tensor_t.shape[3], 1)
    torch.nn.init.normal_(layer_linear.weight, std=0.02) #Initialize the weights to a normal distribution with sd=0.02 (IN THE ORIGINAL CODE, THEY USE TRUNCATED NORMAL DISTRIBUTION)        
    #Do the forward pass
    logits =  layer_linear(token_tensor_t)

    # --> 12 * 385 * 11
    logits = torch.squeeze(logits, dim=-1)
    
    # mask: 12 * 11
    def sequence_mask(lengths, maxlen):
        """
        Returns a mask tensor representing the first n positions of each cell.
        Equivalent to tf.sequence_mask() with param dtype=tf.float32.
        """
        if maxlen is None:
            maxlen = lengths.max()
        mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
        mask = mask.numpy().astype('float32')
        mask = torch.from_numpy(mask)
        return mask

    def flip(x, dim):
        """
        Reverses specific dimensions of a tensor.
        Equivalent to tf.reverse().
        """
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    # mask: 12 * 11 --> after expand_dims: 12 * 1 * 11
    logits_mask = sequence_mask(torch.tensor(slice_mask), args.max_considered_history_turns)
    logits_mask = flip(logits_mask, 1)
    logits_mask = torch.unsqueeze(logits_mask, dim=1)
    exp_logits_masked = torch.exp(logits) * logits_mask

    # --> e.g. 4 * 385 * 11
    exp_logits_masked = exp_logits_masked[:slice_num, :, :]
    probs = exp_logits_masked / torch.sum(exp_logits_masked, dim=2, keepdim=True)
    
    # e.g. 4 * 385 * 11 * 768
    token_tensor_t = token_tensor_t[:slice_num, :, :, :]

    # 4 * 385 * 11 * 1
    probs = torch.unsqueeze(probs, dim=-1)
    
    # 4 * 385 * 768
    new_bert_representation = torch.sum(token_tensor_t * probs, dim=2)

    new_bert_representation, new_mtl_input = splits = torch.split(bert_representation, [args.max_seq_length, 1], 1)

    squeezed_probs = torch.squeeze(probs)
    
    return new_bert_representation, new_mtl_input, squeezed_probs

class MTLLoss():
    def __init__(self, args):
        self.args = args

    def compute_total_loss(self, fd, fd_output, start_logits, end_logits, yesno_logits, followup_logits):
        softmax = torch.nn.Softmax(dim=-1)
        start_probs = softmax(start_logits)
        start_prob = torch.max(start_probs, axis=-1)
        end_probs = softmax(end_logits)
        end_prob = torch.max(end_probs, axis=-1)

        # get the losses - the start loss is for identifying the correct start of the answer span,
        # end loss is identifying the correct end of the answer span

        start_loss = self.compute_cqa_loss(start_logits, fd_output['start_positions'], self.args.max_seq_length)
        end_loss = self.compute_cqa_loss(end_logits, fd_output['end_positions'], self.args.max_seq_length)

        yesno_labels = fd_output['yesno']
        followup_labels = fd_output['followup']

        yesno_loss = torch.mean(self.compute_sparse_softmax_cross_entropy(yesno_logits, yesno_labels))
        followup_loss = torch.mean(self.compute_sparse_softmax_cross_entropy(followup_logits, followup_labels))

        if self.args.do_MTL:
            cqa_loss = (start_loss + end_loss) / 2.0
            if self.args.MTL_lambda < 1:
                total_loss = self.args.MTL_mu * cqa_loss + self.args.MTL_lambda * yesno_loss + \
                             self.args.MTL_lambda * followup_loss
            else:
                total_loss = cqa_loss + yesno_loss + followup_loss
        else:
            total_loss = (start_loss + end_loss) / 2.0

        return total_loss

    def compute_cqa_loss(self, logits, positions, seq_length):
        one_hot_positions = torch.nn.functional.one_hot(positions, seq_length)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(one_hot_positions * log_probs, dim=-1))
        return loss

    def compute_sparse_softmax_cross_entropy(self, logits, labels):
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        logpy = torch.gather(logp, 1, Variable(labels.view(-1, 1)))
        loss = -(logpy).mean()
        return loss
