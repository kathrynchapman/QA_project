from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import tensorflow as tf
from argparse import ArgumentParser
import torch
from transformers import BertPreTrainedModel, BertConfig, BertTokenizer
from bert_model import BertModel


def bert_rep(bert_config, is_training, input_ids, input_mask, segment_ids, history_answer_marker, use_one_hot_embeddings):
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

    model = BertModel(bert_config, )
    inputs = {
        # "config":bert_config,
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "token_type_ids": segment_ids,
        "history_answer_marker": history_answer_marker,
    }

    outputs = model(**inputs)

    sequence_output = outputs[0]  # final hidden layer, with dimensions [batch_size, max_seq_len, hidden_size]
    pooled_output = outputs[1]  # entire sequence representation/embedding of 'CLS' token

    # print("CLS:", final_hidden.shape)
    # print("sent_rep:", sent_rep.shape)
    return sequence_output, pooled_output


def cqa_model(final_hidden):
    """
    :param final_hidden: torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), 
        sequence of hidden-states at the output of the last layer of the model
    :return start_logits: torch.Tensor object
    :return end_logits: torch.Tensor object
    """

    final_hidden_shape = final_hidden.shape
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    #In the original code, they use truncated normal distribution, but there's no function for this in pytorch
    #I found this workaround https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
    #But maybe it's not worth it to go through all that trouble, so I just used normal distribution for now
    output_weights = torch.empty(2, 768).normal_(std=0.02)

    output_bias = torch.zeros(2)

    final_hidden_matrix = final_hidden.view(batch_size * seq_length, hidden_size)
    logits = torch.matmul(final_hidden_matrix, torch.transpose(output_weights, 0, 1))
    logits = torch.add(logits, output_bias)

    logits = logits.reshape(batch_size, seq_length, 2)
    logits = logits.permute(2, 0, 1)
    # logits = logits.T

    unstacked_logits = torch.unbind(logits, dim=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return start_logits, end_logits


def yesno_model(sent_rep):
    """
    :param sent_rep: torch.FloatTensor of shape (batch_size, hidden_size), 
        last layer hidden-state of the first token of the sequence (classification token) further processed by 
        a Linear layer and a Tanh activation function
    :return logits: torch.Tensor object
    """

    linear_layer = torch.nn.Linear(sent_rep.shape[1], 3)

    #Initialize the weights to a normal distribution with sd=0.02 (IN THE ORIGINAL CODE, THEY USE TRUNCATED NORMAL DISTRIBUTION)
    torch.nn.init.normal_(linear_layer.weight, std=0.02)

    logits = linear_layer(sent_rep)

    return logits


def followup_model(sent_rep):
    """
    :param sent_rep: torch.FloatTensor of shape (batch_size, hidden_size),
        last layer hidden-state of the first token of the sequence (classification token) further processed by
        a Linear layer and a Tanh activation function
    :return logits: torch.Tensor object
    """

    linear_layer = torch.nn.Linear(sent_rep.shape[1], 3)

    #Initialize the weights to a normal distribution with sd=0.02 (IN THE ORIGINAL CODE, THEY USE TRUNCATED NORMAL DISTRIBUTION)
    torch.nn.init.normal_(linear_layer.weight, std=0.02)

    logits = linear_layer(sent_rep)

    return logits


def history_attention_net(args, bert_representation, history_attention_input, mtl_input, slice_mask, slice_num):
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
    padding = torch.nn.ZeroPad2d((0, 0, 0, args.batch_size-slice_num)) # padding at the bottom
    history_attention_input = padding(history_attention_input)
    
    # split history_attention_input into 8 tensors of sizes: (3, 4), (3, 4), (2, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)
    splits = torch.split(history_attention_input, slice_mask, 0)

    # pad the splits so that all of them are of size (11, 4)
    def pad_fn(x, num):
        padding = torch.nn.ZeroPad2d((0, 0, args.max_considered_history_turns-num, 0)) # padding at the top
        return padding(x) 
        
    padded = []
    for i in range(args.batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
 
    # stack the splits to form an input_tensor of size (8, 11, 4)
    input_tensor = torch.stack(padded, axis=0)

    # pass input_tensor to a single-layer feed-forward neural network, after which the input_tensor will be of size (8, 11, 1)

    # create network layers
    layer_linear = torch.nn.Linear(input_tensor.shape[2], 1)
    torch.nn.init.normal_(layer_linear.weight, std=0.02) # initialize the weights to a normal distribution (the original TensorFlow code uses a truncated normal distribution)        
    # do the forward pass
    logits =  layer_linear(input_tensor)

    # squeeze input_tensor along dimension 2, so that it has size (8, 11)
    logits = torch.squeeze(logits, dim=2)
    
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