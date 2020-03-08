from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
# from bert import modeling

#import optimization
#import six
import tensorflow as tf
from argparse import ArgumentParser
import transformers
import torch
import numpy #TEST


from transformers import BertPreTrainedModel
# from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BertModel, \
#     BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING
from transformers import BertConfig, BertTokenizer
from bert_model import BertModel

"""
:param bert_config: 'BertConfig' instance
:param input_ids: torch.LongTensor of shape (batch_size, sequence_length),
    indices of input sequence tokens in the vocabulary, which can be obtained using transformers.BertTokenizer
:return final_hidden: torch.FloatTensor of shape (batch_size, sequence_length, hidden_size),
    sequence of hidden-states at the output of the last layer of the model
:return sent_rep: torch.FloatTensor of shape (batch_size, hidden_size),
    last layer hidden-state of the first token of the sequence (classification token) further processed by
    a Linear layer and a Tanh activation function
"""


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
    model = BertModel.from_pretrained('bert-large-cased')
    inputs = {
        # "config":bert_config,
        "input_ids":input_ids,
        "attention_mask":input_mask,
        "token_type_ids":segment_ids,
        "history_answer_marker":history_answer_marker,
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
    #In the cqa_flags.py file, they specify that the value for bert hidden units can be 768 or 1024 and set it to 768
    #I had to set it to 1024 though, otherwise I was getting size mismatch when I tried to do matrix multiplication later
    output_weights = torch.empty(2, 1024).normal_(std=0.02)

    output_bias = torch.zeros(2)

    final_hidden_matrix = final_hidden.view(batch_size * seq_length, hidden_size)
    logits = torch.matmul(final_hidden_matrix, torch.transpose(output_weights, 0, 1))
    logits = torch.add(logits, output_bias)

    logits = logits.view(batch_size, seq_length, 2)
    logits = logits.permute(2, 0, 1)

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


def history_attention_net(args, bert_representation, history_attention_input, mtl_input, slice_mask, slice_num):
    """
    :param bert_representation: torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), 
        sequence of hidden-states at the output of the last layer of the model
    :param history_attention_input: torch.Tensor
    :param mtl_input:
    :param slice_mask: list of size=train_batch_size, containing integers corresponding to the size
        of each subtensor we will get after splitting the history_attention_input tensor
    :param slice_num: int
    :return new_bert_representation: 
    :return new_mtl_input: 
    :return squeezed probs:
    """


    # 12 * 768 --> e.g. 20 * 768
    padding = torch.nn.ZeroPad2d((0, 0, 0, 12-slice_num)) #train_batch_size=12, padding at the bottom
    history_attention_input = padding(history_attention_input)
    
    # the splits contains 12 feature groups. e.g. the first might be 4 * 768 (the number 4 is just an example)
    print(history_attention_input.shape)
    splits = torch.split(history_attention_input, slice_mask, 0)
    print(len(splits))
    print(splits[0].shape)
    print(splits[1].shape)
    # print(splits)
    # --> 11 * 768
    def pad_fn(x, num):
        padding = torch.nn.ZeroPad2d((0, 0, 11-num, 0)) #max_history_turns=11, padding at the top
        return padding(x) 
        
    padded = []
    for i in range(args.batch_size): #train_batch_size=12, but i used 3 for my mini-example
        padded.append(pad_fn(splits[i], slice_mask[i]))
    
    # --> 12 * 11 * 768
    input_tensor = torch.stack(padded, axis=0)
    #I think there is no need to include the following line of code when we use pytorch
    #There's a complete explanation of how tf.tensor.set_shape() works here: 
    #https://stackoverflow.com/questions/35451948/clarification-on-tf-tensor-set-shape
    #input_tensor.set_shape([FLAGS.train_batch_size, FLAGS.max_history_turns, FLAGS.bert_hidden])
#    if FLAGS.history_attention_hidden:
    #Good explanation of dimensions: https://mc.ai/pytorch-layer-dimensions-what-sizes-should-they-be-and-why/
    if True: #TEST, original is with flags
        #Create network layers
        layer_linear1 = torch.nn.Linear(input_tensor.shape[2], 100)
        torch.nn.init.normal_(layer_linear1.weight, std=0.02) #Initialize the weights to a normal distribution with sd=0.02 (IN THE ORIGINAL CODE, THEY USE TRUNCATED NORMAL DISTRIBUTION)
        layer_relu = torch.nn.ReLU()
        layer_linear2 = torch.nn.Linear(100, 1)
        torch.nn.init.normal_(layer_linear2.weight, std=0.02) #Initialize the weights to a normal distribution with sd=0.02 (IN THE ORIGINAL CODE, THEY USE TRUNCATED NORMAL DISTRIBUTION)
        #Do the forward pass
        logits = layer_linear1(input_tensor)
        logits = layer_relu(logits)
        logits = layer_linear2(logits)
    if not True: #TEST, original is with flags
        # --> 12 * 11 * 1
        #Create network layers
        layer_linear = torch.nn.Linear(input_tensor.shape[2], 1)
        torch.nn.init.normal_(layer_linear.weight, std=0.02) #Initialize the weights to a normal distribution with sd=0.02 (IN THE ORIGINAL CODE, THEY USE TRUNCATED NORMAL DISTRIBUTION)        
        #Do the forward pass
        logits =  layer_linear(input_tensor)
    # --> 12 * 11
    logits = torch.squeeze(logits, dim=2)
    
    # mask: 12 * 11
    def sequence_mask(lengths, maxlen):
        """
        Returns a mask tensor representing the first N positions of each cell.
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

    logits_mask = sequence_mask(torch.tensor(slice_mask), 11)


    logits_mask = flip(logits_mask, 1)

    exp_logits_masked = torch.exp(logits) * logits_mask
    # exp_logits_masked = torch.mul(torch.exp(logits) * logits_mask)
    
    # --> e.g. 4 * 11
    exp_logits_masked = exp_logits_masked[:slice_num, :]
    probs = exp_logits_masked / torch.sum(exp_logits_masked, dim=1, keepdim=True)
    
    # e.g. 4 * 11 * 768
    input_tensor = input_tensor[:slice_num, :, :]
    
    # 4 * 11 * 1
    probs = torch.unsqueeze(probs, dim=-1)

    padding = torch.nn.ZeroPad2d((0, 0, 0, 12-slice_num)) ##train_batch_size=12, padding at the bottom
    mtl_input = padding(mtl_input)
    splits = torch.split(mtl_input, slice_mask, 0) 
    padded = []
    for i in range(args.batch_size): #train_batch_size=12, but i used 3 for my mini-example
        padded.append(pad_fn(splits[i], slice_mask[i]))
    mtl_input = torch.stack(padded, axis=0)
    mtl_input = mtl_input[:slice_num, :, :]    
    
    # 4 * 768
    new_mtl_input = torch.sum(mtl_input * probs, dim=1)
    
    # after slicing, the shape information is lost, we rest it
    #I THINK THIS LINE IS NOT NECESSARY IN OUR PYTORCH IMPLEMENTATION
#    new_mtl_input.set_shape([None, FLAGS.bert_hidden])
    
    # 12 * 384 * 768 --> 20 * 384 * 768   
    bert_representation = torch.nn.functional.pad(bert_representation, (0, 0, 0, 0, 0, 12-slice_num)) ##train_batch_size=12, padding at the back
    splits = torch.split(bert_representation, slice_mask, 0) 


    ## from kathryn: there seems to be some sort of dimensional mismatch here? Not sure what's going on...
    pad_fn = lambda x, num: torch.nn.functional.pad(x, (0, 0, 12-num, 0, 0, 0)) #train_batch_size=12, padding at the top

    padded = []
    for i in range(args.batch_size): #train_batch_size=12, but i used 3 for my mini-example
        padded.append(pad_fn(splits[i], slice_mask[i]))

    # for index, tensor in enumerate(padded): #TEST
    #     padded[index] = tensor[:2, :15, :] #TEST

        
    # --> 12 * 11 * 384 * 768
    token_tensor = torch.stack(padded, axis=0)
    # --> 4 * 11 * 384 * 768
    token_tensor = token_tensor[:slice_num, :, :, :]
    
    # 4 * 11 * 1 * 1
    probs = torch.unsqueeze(probs, dim=-1)
    
    # 4 * 384 * 768
    probs = probs[:, :2, :, :] #TEST
    new_bert_representation = torch.sum(token_tensor * probs, dim=1)
#    Following line is not necessary in PyTorch:
#    new_bert_representation.set_shape([None, FLAGS.max_seq_length, FLAGS.bert_hidden])

    squeezed_probs = torch.squeeze(probs)
    
    return new_bert_representation, new_mtl_input, squeezed_probs


def disable_history_attention_net(bert_representation, history_attention_input, mtl_input, slice_mask, slice_num):
    
    # 12 * 768 --> e.g. 20 * 768
    history_attention_input = tf.pad(history_attention_input, [[0, FLAGS.train_batch_size - slice_num], [0, 0]])  
    
    # the splits contains 12 feature groups. e.g. the first might be 4 * 768 (the number 4 is is just an example)
    splits = tf.split(history_attention_input, slice_mask, 0)
    
    # --> 11 * 768
    pad_fn = lambda x, num: tf.pad(x, [[FLAGS.max_history_turns - num, 0], [0, 0]])   
    # padded = tf.map_fn(lambda x: pad_fn(x[0], x[1]), (list(splits), slice_mask), dtype=tf.float32) 
    padded = []
    for i in range(FLAGS.train_batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
    
    # --> 12 * 11 * 768
    input_tensor = tf.stack(padded, axis=0)
    input_tensor.set_shape([FLAGS.train_batch_size, FLAGS.max_history_turns, FLAGS.bert_hidden])
    
#     if FLAGS.history_attention_hidden:
#         hidden = tf.layers.dense(input_tensor, 100, activation=tf.nn.relu,
#                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='history_attention_hidden')
#         logits = tf.layers.dense(hidden, 1, activation=None,
#                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='history_attention_model')
#     else:
#         # --> 12 * 11 * 1
#         logits = tf.layers.dense(input_tensor, 1, activation=None,
#                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='history_attention_model')
#     # --> 12 * 11
#     logits = tf.squeeze(logits, axis=2)
    
    # we assign equal logits, which means equal attention weights. this helps us to see whether the attention net works as expected
    logits = tf.ones((FLAGS.train_batch_size, FLAGS.max_history_turns))

    
    # mask: 12 * 11
    logits_mask = tf.sequence_mask(slice_mask, FLAGS.max_history_turns, dtype=tf.float32)
    logits_mask = tf.reverse(logits_mask, axis=[1])
    exp_logits_masked = tf.exp(logits) * logits_mask 
    
    # --> e.g. 4 * 11
    exp_logits_masked = tf.slice(exp_logits_masked, [0, 0], [slice_num, -1])
    probs = exp_logits_masked / tf.reduce_sum(exp_logits_masked, axis=1, keepdims=True)
    
    # e.g. 4 * 11 * 768
    input_tensor = tf.slice(input_tensor, [0, 0, 0], [slice_num, -1, -1])
    
    # 4 * 11 * 1
    probs = tf.expand_dims(probs, axis=-1)
    
    mtl_input = tf.pad(mtl_input, [[0, FLAGS.train_batch_size - slice_num], [0, 0]]) 
    splits = tf.split(mtl_input, slice_mask, 0)
    pad_fn = lambda x, num: tf.pad(x, [[FLAGS.max_history_turns - num, 0], [0, 0]])   
    padded = []
    for i in range(FLAGS.train_batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
    mtl_input = tf.stack(padded, axis=0)
    mtl_input = tf.slice(mtl_input, [0, 0, 0], [slice_num, -1, -1])
    
    
    # 4 * 768
    new_mtl_input = tf.reduce_sum(mtl_input * probs, axis=1)
    
    # after slicing, the shape information is lost, we rest it
    new_mtl_input.set_shape([None, FLAGS.bert_hidden])
    
    
    # 12 * 384 * 768 --> 20 * 384 * 768
    bert_representation = tf.pad(bert_representation, [[0, FLAGS.train_batch_size - slice_num], [0, 0], [0, 0]])
    splits = tf.split(bert_representation, slice_mask, 0)
    
    pad_fn = lambda x, num: tf.pad(x, [[FLAGS.max_history_turns - num, 0], [0, 0], [0, 0]])
    # padded = tf.map_fn(lambda x: pad_fn(x[0], x[1]), (list(splits), slice_mask), dtype=tf.float32) 
    padded = []
    for i in range(FLAGS.train_batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
        
    # --> 12 * 11 * 384 * 768
    token_tensor = tf.stack(padded, axis=0)
    # --> 4 * 11 * 384 * 768
    token_tensor = tf.slice(token_tensor, [0, 0, 0, 0], [slice_num, -1, -1, -1])
    
    # 4 * 11 * 1 * 1
    probs = tf.expand_dims(probs, axis=-1)
    
    # 4 * 384 * 768
    new_bert_representation = tf.reduce_sum(token_tensor * probs, axis=1)
    new_bert_representation.set_shape([None, FLAGS.max_seq_length, FLAGS.bert_hidden])
    
    return new_bert_representation, new_mtl_input, tf.squeeze(probs)


def fine_grained_history_attention_net(bert_representation, mtl_input, slice_mask, slice_num):
    
    # first concat the bert_representation and mtl_input togenther
    # so that we can process them together
    # shape for bert_representation: 12 * 384 * 768, shape for mtl_input: 12 * 768
    # after concat: 12 * 385 * 768
    
    # 12 * 385 * 768 --> 20 * 385 * 768
    bert_representation = tf.concat([bert_representation, tf.expand_dims(mtl_input, axis=1)], axis=1)
    bert_representation = tf.pad(bert_representation, [[0, FLAGS.train_batch_size - slice_num], [0, 0], [0, 0]])
    splits = tf.split(bert_representation, slice_mask, 0)
    
    pad_fn = lambda x, num: tf.pad(x, [[FLAGS.max_history_turns - num, 0], [0, 0], [0, 0]])
    # padded = tf.map_fn(lambda x: pad_fn(x[0], x[1]), (list(splits), slice_mask), dtype=tf.float32) 
    padded = []
    for i in range(FLAGS.train_batch_size):
        padded.append(pad_fn(splits[i], slice_mask[i]))
        
    # --> 12 * 11 * 385 * 768
    token_tensor = tf.stack(padded, axis=0)
    token_tensor.set_shape([FLAGS.train_batch_size, FLAGS.max_history_turns, FLAGS.max_seq_length + 1, FLAGS.bert_hidden])
    
    # --> 12 * 385 * 11 * 768
    token_tensor_t = tf.transpose(token_tensor, [0, 2, 1, 3])
    
    if FLAGS.history_attention_hidden:
        hidden = tf.layers.dense(token_tensor_t, 100, activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='history_attention_hidden')
        logits = tf.layers.dense(hidden, 1, activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='history_attention_model')
    else:
        # --> 12 * 385 * 11 * 1
        logits = tf.layers.dense(token_tensor_t, 1, activation=None,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='history_attention_model')
    
    # --> 12 * 385 * 11
    logits = tf.squeeze(logits, axis=-1)
    
    # mask: 12 * 11 --> after expand_dims: 12 * 1 * 11
    logits_mask = tf.sequence_mask(slice_mask, FLAGS.max_history_turns, dtype=tf.float32)
    logits_mask = tf.reverse(logits_mask, axis=[1])
    logits_mask = tf.expand_dims(logits_mask, axis=1)
    exp_logits_masked = tf.exp(logits) * logits_mask 
    
    # --> e.g. 4 * 385 * 11
    exp_logits_masked = tf.slice(exp_logits_masked, [0, 0, 0], [slice_num, -1, -1])
    probs = exp_logits_masked / tf.reduce_sum(exp_logits_masked, axis=2, keepdims=True)

    # --> 4 * 385 * 11 * 768
    token_tensor_t = tf.slice(token_tensor_t, [0, 0, 0, 0], [slice_num, -1, -1, -1])
    
    # 4 * 385 * 11 * 1
    probs = tf.expand_dims(probs, axis=-1)
    
    # 4 * 385 * 768
    new_bert_representation = tf.reduce_sum(token_tensor_t * probs, axis=2)
    
    new_bert_representation.set_shape([None, FLAGS.max_seq_length + 1, FLAGS.bert_hidden])
    
    new_bert_representation, new_mtl_input = tf.split(new_bert_representation, [FLAGS.max_seq_length, 1], axis=1)
    new_mtl_input = tf.squeeze(new_mtl_input, axis=1)
    
    return new_bert_representation, new_mtl_input, tf.squeeze(probs)