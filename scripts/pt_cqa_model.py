from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
from bert_model import BertModel
import torch.nn as nn


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=0., b=1.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    a = -std * 2
    b = std * 2
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class CQABertModel(nn.Module):
    """
    The BERT encoder adapted for our implementation to include 'history_answer_marker' input
    """

    def __init__(self, args, config):
        super(CQABertModel, self).__init__()
        self.args = args
        self.model = BertModel(config, self.args)

        # if (self.args.n_gpu > 1 and self.args.device != "cpu"):
        #     self.model = nn.DataParallel(self.model)
        if self.args.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        self.model.to(args.device)

    def forward(self, input_ids, input_mask, segment_ids, history_answer_marker, use_one_hot_embeddings):
        inputs = {
            "input_ids": input_ids.to(self.args.device),
            "attention_mask": input_mask.to(self.args.device),
            "token_type_ids": segment_ids.to(self.args.device),
            "history_answer_marker": history_answer_marker.to(self.args.device),
        }

        outputs = self.model(**inputs)
        sequence_output = outputs[0]  # final hidden layer, with dimensions [batch_size, max_seq_len, hidden_size]
        pooled_output = outputs[1]  # entire sequence representation/embedding of 'CLS' token
        return sequence_output, pooled_output


class CQAModel(nn.Module):
    """
    For answer span prediction
    """

    def __init__(self, args):
        super(CQAModel, self).__init__()
        self.args = args
        self.linear_layer = nn.Linear(self.args.bert_hidden, 2, bias=True)
        torch.nn.init.normal_(self.linear_layer.weight, std=0.02)
        torch.nn.init.zeros_(self.linear_layer.bias)

    def forward(self, final_hidden):
        final_hidden_shape = final_hidden.shape
        batch_size = final_hidden_shape[0]
        seq_length = final_hidden_shape[1]
        hidden_size = final_hidden_shape[2]
        final_hidden_matrix = final_hidden.reshape(batch_size * seq_length, hidden_size)

        logits = self.linear_layer(final_hidden_matrix)

        logits = logits.reshape(batch_size, seq_length, 2)
        logits = logits.permute(2, 0, 1)

        unstacked_logits = torch.unbind(logits, dim=0)

        (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

        return start_logits, end_logits


class YesNoModel(nn.Module):
    """
    For yes/no dialog act prediction
    """

    def __init__(self, args):
        super(YesNoModel, self).__init__()
        self.args = args
        self.linear_layer = nn.Linear(self.args.bert_hidden, 3, bias=False)
        # trunc_normal_(self.linear_layer.weight, std=0.02)
        torch.nn.init.normal_(self.linear_layer.weight, std=0.02)

    def forward(self, sent_rep):
        logits = self.linear_layer(sent_rep)
        return logits


class FollowUpModel(nn.Module):
    """
    For follow-up dialog act prediction
    """

    def __init__(self, args):
        super(FollowUpModel, self).__init__()
        self.args = args
        self.linear_layer = nn.Linear(self.args.bert_hidden, 3, bias=False)
        # trunc_normal_(self.linear_layer.weight, std=0.02)
        torch.nn.init.normal_(self.linear_layer.weight, std=0.02)

    def forward(self, sent_rep):
        logits = self.linear_layer(sent_rep)
        return logits


class HistoryAttentionNet(nn.Module):
    """
    History attention net, performed at sequence-level
    """

    def __init__(self, args):
        super(HistoryAttentionNet, self).__init__()
        self.args = args
        self.layer_linear = nn.Linear(self.args.bert_hidden, 1, bias=False)
        # trunc_normal_(self.layer_linear.weight, std=0.02)
        torch.nn.init.normal_(self.layer_linear.weight, std=0.02)

    def pad_fn(self, x, num):
        # pad the splits so that all of them are of size (11, 4)
        padding = nn.ZeroPad2d((0, 0, self.args.max_considered_history_turns - num, 0))  # padding at the top
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

    def forward(self, bert_representation, mtl_input, slice_mask, slice_num, history_attention_input):
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
        # pad history_attention_input: (8, 3) --> (13, 3)
        padding = nn.ZeroPad2d((0, 0, 0, self.args.batch_size - slice_num))  # padding at the bottom
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
        # exp_logits_masked = torch.exp(logits) * logits_mask.to('cuda:0')
        exp_logits_masked = torch.exp(logits) * logits_mask.to(self.args.device)
        # use the softmax function to generate a tensor of probabilities
        # slice the resulting exp_logits_masked tensor to get rid of the padded rows, obtaining a tensor of size (3, 11)
        exp_logits_masked = exp_logits_masked[:slice_num, :]
        # divide each cell by the sum of all cells
        probs = exp_logits_masked / torch.sum(exp_logits_masked, dim=1, keepdim=True)

        # unsqueeze the probs tensor so that it has size (3, 11, 1)
        probs = torch.unsqueeze(probs, dim=-1)

        #### GENERATE TENSOR: new_mtl_input ####

        # pad mtl_input: (8, 3) --> (13, 3)
        padding = nn.ZeroPad2d((0, 0, 0, self.args.batch_size - slice_num))  # padding at the bottom
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
                                                      (0, 0, 0, 0, 0,
                                                       self.args.batch_size - slice_num))  # padding at the back

        # split mtl_input into tensors of sizes: (3, 12, 4), (3, 12, 4), (2, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4)
        splits = torch.split(bert_representation, slice_mask, 0)

        # pad the splits so that all of them are of size (11, 12, 4)
        pad_fn = lambda x, num: nn.functional.pad(x, (
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


class DisableHistoryAttentionNet(nn.Module):
    def __init__(self, args):
        super(DisableHistoryAttentionNet, self).__init__()
        self.args = args

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

    def pad_fn(self, x, num):
        padding = nn.ZeroPad2d((0, 0, self.args.max_considered_history_turns - num, 0))  # padding at the top
        return padding(x)

    def forward(self, bert_representation, mtl_input, slice_mask, slice_num, history_attention_input):
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
        padding = nn.ZeroPad2d((0, 0, 0, self.args.batch_size - slice_num))  # padding at the bottom
        history_attention_input = padding(history_attention_input)

        # split history_attention_input into 8 tensors of sizes: (3, 4), (3, 4), (2, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)
        splits = torch.split(history_attention_input, slice_mask, 0)

        # pad the splits so that all of them are of size (11, 4)

        padded = []
        for i in range(self.args.batch_size):
            padded.append(self.pad_fn(splits[i], slice_mask[i]))

        # stack the splits to form an input_tensor of size (8, 11, 4)
        input_tensor = torch.stack(padded, axis=0)

        # we assign equal logits, which means equal attention weights
        # this helps us to see whether the attention networks as expected
        logits = torch.ones(self.args.batch_size, self.args.max_considered_history_turns)

        logits_mask = self.sequence_mask(torch.tensor(slice_mask), self.args.max_considered_history_turns)
        logits_mask = self.flip(logits_mask, 1)
        exp_logits_masked = torch.exp(logits) * logits_mask

        # use the softmax function to generate a tensor of probabilities
        # slice the resulting exp_logits_masked tensor to get rid of the padded rows, obtaining a tensor of size (3, 11)
        exp_logits_masked = exp_logits_masked[:slice_num, :]
        # divide each cell by the sum of all cells
        probs = exp_logits_masked / torch.sum(exp_logits_masked, dim=1, keepdim=True)

        # slice input_tensor to get rid of the paddings, resulting in a tensor of size (3, 11, 4)
        input_tensor = input_tensor[:slice_num, :, :]

        # unsqueeze the probs tensor so that it has size (3, 11, 1)
        probs = torch.unsqueeze(probs, dim=-1)

        #### GENERATE TENSOR: new_mtl_input ####

        # pad mtl_input: (8, 3) --> (13, 3)
        padding = nn.ZeroPad2d((0, 0, 0, self.args.batch_size - slice_num))  # padding at the bottom
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
        new_mtl_input = torch.sum(mtl_input * probs.to(self.args.device), dim=1)

        #### GENERATE TENSOR: new_bert_representation ####

        # pad bert_representation: (8, 12, 4) --> (13, 12, 4)
        bert_representation = nn.functional.pad(bert_representation, (
            0, 0, 0, 0, 0, self.args.batch_size - slice_num))  # padding at the back

        # split mtl_input into tensors of sizes: (3, 12, 4), (3, 12, 4), (2, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4), (1, 12, 4)
        splits = torch.split(bert_representation, slice_mask, 0)

        # pad the splits so that all of them are of size (11, 12, 4)
        pad_fn = lambda x, num: nn.functional.pad(x, (
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
        new_bert_representation = torch.sum(token_tensor.to(self.args.device) * probs.to(self.args.device), dim=1)

        # squeeze back the probs tensor: (3, 11, 1, 1) --> (3, 11, 1)
        probs = torch.squeeze(probs)

        return new_bert_representation, new_mtl_input, probs


class FineGrainedHistoryAttentionNet(nn.Module):
    """
    History attention net, performed at token-level
    """

    def __init__(self, args):
        super(FineGrainedHistoryAttentionNet, self).__init__()
        self.args = args
        self.layer_linear = nn.Linear(self.args.bert_hidden, 1, bias=False)
        # trunc_normal_(self.layer_linear.weight, std=0.02)
        torch.nn.init.normal_(self.layer_linear.weight, std=0.02)

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

    def forward(self, bert_representation, mtl_input, slice_mask, slice_num, history_attention_input=None):
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
        history_attention_input = None
        bert_representation = torch.cat((bert_representation, torch.unsqueeze(mtl_input, dim=1)), dim=1)

        # pad bert_representation: (8, 13, 4) --> (13, 13, 4)
        bert_representation = nn.functional.pad(bert_representation, (
            0, 0, 0, 0, 0, self.args.batch_size - slice_num))  # padding at the back

        # split bert_representation into 8 tensors of sizes: (3, 13, 4), (3, 13, 4), (2, 13, 4), (1, 13, 4), (1, 13, 4), (1, 13, 4), (1, 13, 4), (1, 13, 4)
        splits = torch.split(bert_representation, slice_mask, 0)

        # pad the splits so that all of them are of size (11, 13, 4)
        pad_fn = lambda x, num: nn.functional.pad(x, (
            0, 0, 0, 0, self.args.max_considered_history_turns - num, 0))  # padding at the front
        padded = []
        for i in range(self.args.batch_size):
            padded.append(pad_fn(splits[i], slice_mask[i]))

        # stack the splits to form a token_tensor of size (8, 11, 13, 4)
        token_tensor = torch.stack(padded, axis=0)
        token_tensor.reshape(self.args.batch_size, self.args.max_considered_history_turns, self.args.max_seq_length + 1,
                             self.args.bert_hidden)

        # permute dimensions in token_tensor: (8, 13, 11, 4)
        token_tensor_t = token_tensor.permute(0, 2, 1, 3)

        logits = self.layer_linear(token_tensor_t)

        # squeeze token_tensor along dimension 2, so that it has size (8, 13, 11)
        logits = torch.squeeze(logits, dim=-1)

        # mask the padded parts of token_tensor out and apply the exponential function to all its cells

        logits_mask = self.sequence_mask(torch.tensor(slice_mask), self.args.max_considered_history_turns)
        logits_mask = self.flip(logits_mask, 1)
        logits_mask = torch.unsqueeze(logits_mask, dim=1)

        # use the softmax function to generate a tensor of probabilities
        exp_logits_masked = torch.exp(logits) * logits_mask.to(self.args.device)

        # slice the resulting exp_logits_masked tensor to get rid of the padded rows, obtaining a tensor of size (3, 13, 11)
        exp_logits_masked = exp_logits_masked[:slice_num, :, :]

        # divide each cell by the sum of all cells, resulting in a probs tensor of shape (3, 13, 11)
        probs = exp_logits_masked / torch.sum(exp_logits_masked, dim=2, keepdim=True)

        # slice token_tensor to get rid of the padded part, resulting in a token_tensor_t tensor of size (3, 11, 13, 4)
        token_tensor_t = token_tensor_t[:slice_num, :, :, :]

        # unsqueeze the probs tensor: (3, 13, 11) --> (3, 13, 11, 1)
        probs = torch.unsqueeze(probs, dim=-1)

        # multiply token_tensor_t by probs and sum along dimension 1, resulting in a new_bert_representation tensor of shape (3, 13, 4)
        new_bert_representation = torch.sum(token_tensor_t * probs, dim=2)

        new_bert_representation.reshape(slice_num, self.args.max_seq_length + 1, self.args.bert_hidden)

        # split the new_bert_representation tensor along dimension 1 to get the token-level tensor new_bert_representation (3, 12, 4)
        # and the sequence-level tensor new_mtl_input back (3, 1, 4)
        new_bert_representation, new_mtl_input = torch.split(new_bert_representation, [self.args.max_seq_length, 1], 1)
        new_mtl_input = torch.squeeze(new_mtl_input, axis=1)

        # squeeze back the probs tensor: (3, 13, 11, 1) --> (3, 13, 11)
        squeezed_probs = torch.squeeze(probs)

        return new_bert_representation, new_mtl_input, squeezed_probs


class MTLModel(nn.Module):
    """
    Full multi-task learning model
    Includes BERT encoder, History Attention Model, Conversational QA Model, Yes-No Model, and Follow-up Model
    """

    def __init__(self, args):
        super(MTLModel, self).__init__()
        self.args = args
        self.bert_model = CQABertModel(self.args, self.args.bert_config)
        self.cqa = CQAModel(self.args).to(self.args.device)
        if self.args.do_MTL:
            self.yesno = YesNoModel(self.args).to(self.args.device)
            self.followup = FollowUpModel(self.args).to(self.args.device)
        if self.args.fine_grained_attention:
            self.ham = FineGrainedHistoryAttentionNet(self.args).to(self.args.device)
        elif self.args.disable_attention:
            self.ham = DisableHistoryAttentionNet(self.args).to(self.args.device)
        else:
            self.ham = HistoryAttentionNet(self.args).to(args.device)

    def forward(self, fd, batch_slice_mask, batch_slice_num):
        # encode everything with the BERT model
        bert_representation, cls_representation = self.bert_model(input_ids=fd['input_ids'],
                                                                  input_mask=fd['input_mask'],
                                                                  segment_ids=fd['segment_ids'],
                                                                  history_answer_marker=fd['history_answer_marker'],
                                                                  use_one_hot_embeddings=True)

        reduce_mean_representation = torch.mean(bert_representation, 1)
        history_attention_input = reduce_mean_representation
        mtl_input = reduce_mean_representation

        # apply the HAM
        new_bert_representation, new_mtl_input, attention_weights = self.ham(bert_representation.to(self.args.device),
                                                                             mtl_input.to(self.args.device),
                                                                             batch_slice_mask,
                                                                             batch_slice_num,
                                                                             history_attention_input.to(
                                                                                 self.args.device))

        if self.args.do_MTL:
            return (self.cqa(new_bert_representation), self.yesno(new_mtl_input), self.followup(new_mtl_input),
                    attention_weights)
        else:
            return (self.cqa(new_bert_representation), attention_weights)


class MTLLoss():
    """
    Computes the multi-task learning loss
    """

    def __init__(self, args):
        self.args = args

    def compute_total_loss(self, fd_output, start_logits, end_logits, yesno_logits=None, followup_logits=None):
        """
        Computes the full loss for the multiple models
        :param fd_output: the feed dict for the desired outputs
        :param start_logits: logits for the identifying the start of an answer span
        :param end_logits: logits for the identifying the end of an answer span
        :param yesno_logits: logits for the yesno model
        :param followup_logits: logits for the followup model
        :return: loss from the desired models
        """
        # get the losses - the start loss is for identifying the correct start of the answer span,
        # end loss is identifying the correct end of the answer span

        start_loss = self.compute_cqa_loss(start_logits, fd_output['start_positions'], self.args.max_seq_length)
        end_loss = self.compute_cqa_loss(end_logits, fd_output['end_positions'], self.args.max_seq_length)

        if self.args.do_MTL:
            cqa_loss = (start_loss + end_loss) / 2.0
            yesno_labels = fd_output['yesno']
            followup_labels = fd_output['followup']

            yesno_loss = torch.mean(self.compute_sparse_softmax_cross_entropy(yesno_logits, yesno_labels))
            followup_loss = torch.mean(self.compute_sparse_softmax_cross_entropy(followup_logits, followup_labels))

            if self.args.MTL_lambda < 1:
                total_loss = self.args.MTL_mu * cqa_loss + self.args.MTL_lambda * yesno_loss + \
                             self.args.MTL_lambda * followup_loss
            else:
                total_loss = cqa_loss + yesno_loss + followup_loss
        else:
            total_loss = (start_loss + end_loss) / 2.0

        return total_loss

    def compute_cqa_loss(self, logits, positions, seq_length):
        """
        Computes the loss for the answer span prediction
        :param logits: Final logit outputs from CQA model
        :param positions: Target positions
        :param seq_length: Max sequence length
        :return: loss
        """
        one_hot_positions = nn.functional.one_hot(positions, seq_length)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(one_hot_positions * log_probs, dim=-1))

        return loss

    def compute_sparse_softmax_cross_entropy(self, logits, labels):
        """
        Computes the loss for the dialog act prediction/follow-up
        :param logits: Final logit outputs from YesNoModel/FollowupModel
        :param labels: Correct labels
        :return: loss
        """
        # logp = nn.functional.log_softmax(logits, dim=-1)
        # logpy = torch.gather(logp, 1, Variable(labels.view(-1, 1)))
        # loss = -(logpy).mean()

        loss_fnct = nn.CrossEntropyLoss(reduce='mean')
        loss = loss_fnct(logits, labels)

        return loss
