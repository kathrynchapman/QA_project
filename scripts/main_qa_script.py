# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.tensorboard import SummaryWriter

import collections
import random
import json
import math
import os
# from bert import modeling
# import optimization
import six

import numpy as np
from copy import deepcopy
import pickle
import itertools
from time import time
import traceback
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import modeling_bert, BertConfig, BertTokenizer
from pt_cqa_supports import *
# from cqa_flags import FLAGS
from pt_cqa_model import *
from pt_cqa_gen_batches import cqa_gen_example_aware_batches_v2
# from cqa_rl_supports import *
from scorer import external_call  # quac official evaluation script


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model=None, tokenizer=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    print(train_dataloader[0])


if __name__ == '__main__':

    parser = ArgumentParser(
        description='QA model')

    parser.add_argument("--cache_dir", default=None, required=True, type=str,
                        help="Where the cached data is (to be) stored.")
    parser.add_argument("--output_dir", default=None, required=True, type=str,
                        help="Where the model output data is (to be) stored.")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Whether to overwrite the outputs directory")
    parser.add_argument("--quac_data_dir", default=None, type=str, help="The input data directory.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument('--do_predict', action="store_true", help='Whether to predict or not.')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--train_steps", type=int, default=20,
                        help="loss: the loss gap on reward set, f1: the f1 on reward set")
    parser.add_argument("--load_small_portion", action="store_true", help="Load a small portion of data during dev.")
    parser.add_argument("--max_seq_length", default=384, type=int, help="The maximum total input sequence "
                                                                        "length after tokenization.")
    parser.add_argument("--doc_stride", default=128, type=int, help="When splitting up a long document into chunks,"
                                                                    "how much stride to take between chunks.")
    parser.add_argument("--max_considered_history_turns", default=11, type=int, help="we only consider k history turns "
                                                                                     "that immediately proceed the current turn, when generating preprocessed features,")
    parser.add_argument("--warmup_proportion", default=0.0, help="Proportion of training to perform linear "
                                                                 "learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument("--history_attention_input", default='reduce_mean', type=str,
                        help="CLS, reduce_mean, reduce_max")
    parser.add_argument("--mtl_input", default="reduce_mean", type=str, help="CLS, reduce_mean, reduce_max")
    parser.add_argument("--aux_shared", default=False, type=bool,
                        help="wheter to share the aux prediction layer with the main convqa model")
    parser.add_argument("--disable_attention", action="store_true", help="disable the history attention module")
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size for training and predicting")
    parser.add_argument("--num_epochs", default=1, type=int, help="Number of training epochs")
    args = parser.parse_args()
    args.output_dir = args.output_dir + '/' if args.output_dir[-1] != '/' else args.output_dir

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # set the seed for initialization
    set_seed(args)
    # get the BERT config using huggingface
    bert_config = BertConfig.from_pretrained('bert-base-uncased')

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    # make the cache dir if it doesn't exist
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir + '/summaries/train/')
        os.makedirs(args.output_dir + '/summaries/val/')
        os.makedirs(args.output_dir + '/summaries/rl/')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if args.do_train:
        # read in training data, generate training features, and generate training batches
        train_examples = None
        num_train_steps = None
        num_warmup_steps = None
        train_file = args.quac_data_dir + 'val_v0.2.json' if args.quac_data_dir[
                                                                 -1] == '/' else args.quac_data_dir + '/val_v0.2.json'  # CHANGED TO VAL ONLY FOR TESTING
        if args.load_small_portion:
            train_examples = read_quac_examples(input_file=train_file, is_training=True)[:3]
        else:
            train_examples = read_quac_examples(input_file=train_file, is_training=True)

        features_fname = args.cache_dir + '/train_features_{}_{}.pkl'.format(args.load_small_portion,
                                                                             args.max_considered_history_turns)
        example_tracker_fname = args.cache_dir + '/example_tracker_{}_{}.pkl'.format(args.load_small_portion,
                                                                                     args.max_considered_history_turns)
        variation_tracker_fname = args.cache_dir + '/variation_tracker_{}_{}.pkl'.format(args.load_small_portion,
                                                                                         args.max_considered_history_turns)
        example_features_nums_fname = args.cache_dir + '/example_features_nums_{}_{}.pkl'.format(
            args.load_small_portion, args.max_considered_history_turns)
        try:
            print('attempting to load train features from cache')
            with open(features_fname, 'rb') as handle:
                train_features = pickle.load(handle)
            with open(example_tracker_fname, 'rb') as handle:
                example_tracker = pickle.load(handle)
            with open(variation_tracker_fname, 'rb') as handle:
                variation_tracker = pickle.load(handle)
            with open(example_features_nums_fname, 'rb') as handle:
                example_features_nums = pickle.load(handle)
        except:
            print('train feature cache does not exist, generating')
            train_features, example_tracker, variation_tracker, example_features_nums = convert_examples_to_variations_and_then_features(
                examples=train_examples, tokenizer=tokenizer,
                max_seq_length=args.max_seq_length, doc_stride=args.doc_stride,
                max_query_length=64,
                max_considered_history_turns=args.max_considered_history_turns,
                is_training=True)
            with open(features_fname, 'wb') as handle:
                pickle.dump(train_features, handle)
            with open(example_tracker_fname, 'wb') as handle:
                pickle.dump(example_tracker, handle)
            with open(variation_tracker_fname, 'wb') as handle:
                pickle.dump(variation_tracker, handle)
            with open(example_features_nums_fname, 'wb') as handle:
                pickle.dump(example_features_nums, handle)
            print('train features generated')

        train_batches = cqa_gen_example_aware_batches_v2(train_features, example_tracker, variation_tracker,
                                                         example_features_nums,
                                                         batch_size=args.batch_size, num_epoches=1, shuffle=False)
        temp_train_batches = cqa_gen_example_aware_batches_v2(train_features, example_tracker, variation_tracker,
                                                              example_features_nums,
                                                              batch_size=args.batch_size, num_epoches=1, shuffle=True)
        print('len temp_train_batches', len(list(temp_train_batches)))

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        train_iterator = trange(
            epochs_trained, int(args.num_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
        )

        set_seed(args)  # Added here for reproductibility
        for _ in train_iterator:
            epoch_iterator = tqdm(train_batches, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                batch_features, batch_slice_mask, batch_slice_num, output_features = batch
#                input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
#                    0)  # TEST

                fd = convert_features_to_feed_dict(batch_features)  # feed_dict
                fd_output = convert_features_to_feed_dict(output_features)

                ## the below is just for my reference to know what's in the feed dict - kathryn

                # feed_dict = {'unique_ids': batch_unique_ids, 'input_ids': batch_input_ids,
                #              'input_mask': batch_input_mask, 'segment_ids': batch_segment_ids,
                #              'start_positions': batch_start_positions, 'end_positions': batch_end_positions,
                #              'history_answer_marker': batch_history_answer_marker, 'yesno': batch_yesno,
                #              'followup': batch_followup,
                #              'metadata': batch_metadata}


                bert_representation, cls_representation = bert_rep(bert_config, is_training=True,
                                                                   input_ids=fd['input_ids'],
                                                                   input_mask=fd['input_mask'],
                                                                   segment_ids=fd['segment_ids'],
                                                                   history_answer_marker=fd['history_answer_marker'],
                                                                   use_one_hot_embeddings=True)

                reduce_mean_representation = torch.mean(bert_representation, 1)
                history_attention_input = reduce_mean_representation
                mtl_input = reduce_mean_representation
                (aux_start_logits, aux_end_logits) = cqa_model(bert_representation)
                new_bert_representation, new_mtl_input, attention_weights = history_attention_net(args, bert_representation,
                                                                                                  history_attention_input,
                                                                                                  mtl_input,
                                                                                                  batch_slice_mask,
                                                                                                  batch_slice_num)                                                                                                 
                # yesno_logits = yesno_model(new_mtl_input)
                # followup_logits = followup_model(new_mtl_input)
                # (start_logits, end_logits) = cqa_model(new_bert_representation)


#                print("Done with batch", step)

                # a = torch.ten



        # an estimation of num_train_steps
        # num_train_steps = int((math.ceil(len(train_examples) * 1.5 / FLAGS.train_batch_size)) * FLAGS.num_train_epochs)
        # we cannot predict the exact training steps because of the "example-aware" batching,
        # so we run some initial experiments and found out the exact training steps
        # num_train_steps = 11438 * FLAGS.num_train_epochs
#        num_train_steps = args.train_steps
#        num_warmup_steps = int(num_train_steps * args.warmup_proportion)

#    if args.do_predict:
        # read in validation data, generate val features
#        val_file = args.quac_data_dir + 'val_v0.2.json' if args.quac_data_dir[
#                                                               -1] == '/' else args.quac_data_dir + '/val_v0.2.json'
#        if args.load_small_portion:
#            val_examples = read_quac_examples(input_file=val_file, is_training=False)[
#                           :10]  # USE THE WHOLE DATASET LATER
#        else:
#            val_examples = read_quac_examples(input_file=val_file, is_training=False)

        # we read in the val file in json for the external_call function in the validation step
#        val_file_json = json.load(open(val_file, 'r'))['data']

        # we attempt to read features from cache
#        features_fname = args.cache_dir + '/val_features_{}_{}.pkl'.format(args.load_small_portion,
#                                                                           args.max_considered_history_turns)
#        example_tracker_fname = args.cache_dir + '/val_example_tracker_{}_{}.pkl'.format(args.load_small_portion,
#                                                                                         args.max_considered_history_turns)
#        variation_tracker_fname = args.cache_dir + '/val_variation_tracker_{}_{}.pkl'.format(args.load_small_portion,
#                                                                                             args.max_considered_history_turns)
#        example_features_nums_fname = args.cache_dir + '/val_example_features_nums_{}_{}.pkl'.format(
#            args.load_small_portion, args.max_considered_history_turns)

#        try:
#            print('attempting to load val features from cache')
#            with open(features_fname, 'rb') as handle:
#                val_features = pickle.load(handle)
#            with open(example_tracker_fname, 'rb') as handle:
#                val_example_tracker = pickle.load(handle)
#            with open(variation_tracker_fname, 'rb') as handle:
#                val_variation_tracker = pickle.load(handle)
#            with open(example_features_nums_fname, 'rb') as handle:
#                val_example_features_nums = pickle.load(handle)
#        except:
#            print('val feature cache does not exist, generating')
#            val_features, val_example_tracker, val_variation_tracker, val_example_features_nums = \
#                convert_examples_to_variations_and_then_features(
#                    examples=val_examples, tokenizer=tokenizer,
#                    max_seq_length=args.max_seq_length, doc_stride=args.doc_stride,
#                    max_query_length=64,
#                    max_considered_history_turns=args.max_considered_history_turns,
#                    is_training=False)

#            with open(features_fname, 'wb') as handle:
#                pickle.dump(val_features, handle)
#            with open(example_tracker_fname, 'wb') as handle:
#                pickle.dump(val_example_tracker, handle)
#            with open(variation_tracker_fname, 'wb') as handle:
#                pickle.dump(val_variation_tracker, handle)
#            with open(example_features_nums_fname, 'wb') as handle:
#                pickle.dump(val_example_features_nums, handle)
#            print('val features generated')

#        num_val_examples = len(val_examples)

    # PYTORCH DOES NOT USE PLACEHOLDERS AS TENSORFLOW, WE HAVE TO FIND A WAY OF ADAPTING THIS BLOCK OF CODE FOR PYTORCH.
    # GOOD EXPLANATION: Tensorflow works on a static graph concept that means the user first has to define the computation
    # graph of the model and then run the ML model, whereas PyTorch believes in a dynamic graph that allows defining/manipulating
    # the graph on the go. PyTorch offers an advantage with its dynamic nature of creating the graphs.
    # This placeholders approach is actually from an old version of TensorFlow, the new one is more similar to Pytorch.
    # Because it is a deprecated version of TensorFlow, I had to add the compat.v1 line at the beginning of this block of code to make it work.
    # Even so, I kept running into compatibility issues and in the end I just gave up and relied on the documentation of tensorflow,
    # bert and pytorch to make something equivalent. I was not able to run it both with tensorflow (original) and pytorch (my version)
    # to check if the output is exactly the same.

    # tf Graph input
    #    tf.compat.v1.disable_eager_execution()
    #    unique_ids = tf.placeholder(tf.int32, shape=[None], name='unique_ids')
    #    input_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, 384], name='input_ids')
    #    input_mask = tf.compat.v1.placeholder(tf.int32, shape=[None, 384], name='input_mask')
    #    segment_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, 384], name='segment_ids')
    #    start_positions = tf.placeholder(tf.int32, shape=[None], name='start_positions')
    #    end_positions = tf.placeholder(tf.int32, shape=[None], name='end_positions')
    #    history_answer_marker = tf.compat.v1.placeholder(tf.int32, shape=[None, 384], name='history_answer_marker')
    #    training = tf.compat.v1.placeholder(tf.int32, shape=[None, 384], name='training')
    #    get_segment_rep = tf.placeholder(tf.bool, name='get_segment_rep')
    #    yesno_labels = tf.placeholder(tf.int32, shape=[None], name='yesno_labels')
    #    followup_labels = tf.placeholder(tf.int32, shape=[None], name='followup_labels')

    # a unique combo of (e_tracker, f_tracker) is called a slice
    #    slice_mask = tf.placeholder(tf.int32, shape=[FLAGS.train_batch_size, ], name='slice_mask')
    #    slice_num = tf.placeholder(tf.int32, shape=None, name='slice_num')

    # for auxiliary loss
    #    aux_start_positions = tf.placeholder(tf.int32, shape=[None], name='aux_start_positions')
    #    aux_end_positions = tf.placeholder(tf.int32, shape=[None], name='aux_end_positions')

#    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)# TEST
#    print(input_ids)
#    print(type(input_ids))
#    bert_representation, cls_representation = bert_rep(bert_config, input_ids)

    # for batch in train_batches: #TEST
    #     batch_features, batch_slice_mask, batch_slice_num, output_features = batch #TEST
    #     break #TEST
    # history_attention_net(bert_representation, cls_representation, cls_representation, [4, 5, 2], 2) #TEST

#    reduce_mean_representation = torch.mean(bert_representation, 1)
    #    reduce_max_representation = tf.reduce_max(bert_representation, axis=1)

    #    if FLAGS.history_attention_input == 'CLS':
    #        history_attention_input = cls_representation
#    if args.history_attention_input == 'reduce_mean':
#        history_attention_input = reduce_mean_representation
    #    elif FLAGS.history_attention_input == 'reduce_max':
    #        history_attention_input = reduce_max_representation
    #    else:
    #        print('FLAGS.history_attention_input not specified')

    #    if FLAGS.mtl_input == 'CLS':
    #        mtl_input = cls_representation
#    if args.mtl_input == 'reduce_mean':
#        mtl_input = reduce_mean_representation
    #    elif FLAGS.mtl_input == 'reduce_max':
    #        mtl_input = reduce_max_representation
    #    else:
    #        print('FLAGS.mtl_input not specified')

    # if args.aux_shared:
    #     # if the aux prediction layer is shared with the main convqa model:
    #     (aux_start_logits, aux_end_logits) = cqa_model(bert_representation)
    # else:
    #     # if they are not shared
    #     (aux_start_logits, aux_end_logits) = aux_cqa_model(bert_representation)

#    (aux_start_logits, aux_end_logits) = cqa_model(bert_representation)

    #    if FLAGS.disable_attention:
    #        new_bert_representation, new_mtl_input, attention_weights = disable_history_attention_net(bert_representation,
    #                                                                                        history_attention_input, mtl_input,
    #                                                                                        slice_mask,
    #                                                                                        slice_num)

    #    else:
    #        if FLAGS.fine_grained_attention:
    #            new_bert_representation, new_mtl_input, attention_weights = fine_grained_history_attention_net(bert_representation,
    #                                                                                            mtl_input,
    #                                                                                            slice_mask,
    #                                                                                            slice_num)

    #        else:
#    slice_mask = [0] * args.batch_size
#    slice_num = 0
    # new_bert_representation, new_mtl_input, attention_weights = history_attention_net(bert_representation,
    #                                                                                   history_attention_input,
    #                                                                                   mtl_input,
    #                                                                                   slice_mask,
    #                                                                                   slice_num)

#    (start_logits, end_logits) = cqa_model(new_bert_representation)
#    yesno_logits = yesno_model(new_mtl_input)
#    followup_logits = followup_model(new_mtl_input)

#    tvars = tf.trainable_variables()
# print(tvars)

#    initialized_variable_names = {}
#    if FLAGS.init_checkpoint:
#        (assignment_map, initialized_variable_names) = modeling.get_assigment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
#        tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
# print('tvars',tvars)
# print('initialized_variable_names',initialized_variable_names)

# compute loss
#    seq_length = modeling.get_shape_list(input_ids)[1]
#    def compute_loss(logits, positions):
#        one_hot_positions = tf.one_hot(
#            positions, depth=seq_length, dtype=tf.float32)
#        log_probs = tf.nn.log_softmax(logits, axis=-1)
#        loss = -tf.reduce_mean(tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
#        return loss

# get the max prob for the predicted start/end position
#    start_probs = tf.nn.softmax(start_logits, axis=-1)
#    start_prob = tf.reduce_max(start_probs, axis=-1)
#    end_probs = tf.nn.softmax(end_logits, axis=-1)
#    end_prob = tf.reduce_max(end_probs, axis=-1)

#    start_loss = compute_loss(start_logits, start_positions)
#    end_loss = compute_loss(end_logits, end_positions)

#    yesno_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yesno_logits, labels=yesno_labels))
#    followup_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=followup_logits, labels=followup_labels))

#    if FLAGS.MTL:
#        cqa_loss = (start_loss + end_loss) / 2.0
#        if FLAGS.MTL_lambda < 1:
#            total_loss = FLAGS.MTL_mu * cqa_loss +                      FLAGS.MTL_lambda * yesno_loss +                      FLAGS.MTL_lambda * followup_loss
#        else:
#            total_loss = cqa_loss + yesno_loss + followup_loss
#        tf.summary.scalar('cqa_loss', cqa_loss)
#        tf.summary.scalar('yesno_loss', yesno_loss)
#        tf.summary.scalar('followup_loss', followup_loss)
#    else:
#        total_loss = (start_loss + end_loss) / 2.0


# if FLAGS.aux:
#     aux_start_probs = tf.nn.softmax(aux_start_logits, axis=-1)
#     aux_start_prob = tf.reduce_max(aux_start_probs, axis=-1)
#     aux_end_probs = tf.nn.softmax(aux_end_logits, axis=-1)
#     aux_end_prob = tf.reduce_max(aux_end_probs, axis=-1)
#     aux_start_loss = compute_loss(aux_start_logits, aux_start_positions)
#     aux_end_loss = compute_loss(aux_end_logits, aux_end_positions)

#     aux_loss = (aux_start_loss + aux_end_loss) / 2.0
#     cqa_loss = (start_loss + end_loss) / 2.0
#     total_loss = (1 - FLAGS.aux_lambda) * cqa_loss + FLAGS.aux_lambda * aux_loss

#     tf.summary.scalar('cqa_loss', cqa_loss)
#     tf.summary.scalar('aux_loss', aux_loss)

# else:
#     total_loss = (start_loss + end_loss) / 2.0


#    tf.summary.scalar('total_loss', total_loss)

#    if FLAGS.do_train:
#        train_op = optimization.create_optimizer(total_loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, False)

#        print("***** Running training *****")
#        print("  Num orig examples = %d", len(train_examples))
#        print("  Num train_features = %d", len(train_features))
#        print("  Batch size = %d", FLAGS.train_batch_size)
#        print("  Num steps = %d", num_train_steps)

#    merged_summary_op = tf.summary.merge_all()

#    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits", "yesno_logits", "followup_logits"])

#    attention_dict = {}

#    saver = tf.train.Saver()
# Initializing the variables
#    init = tf.global_variables_initializer()
#    tf.get_default_graph().finalize()
#    with tf.Session() as sess:
#        sess.run(init)

#        if FLAGS.do_train:
#            train_summary_writer = tf.summary.FileWriter(FLAGS.output_dir + 'summaries/train', sess.graph)
#            val_summary_writer = tf.summary.FileWriter(FLAGS.output_dir + 'summaries/val')
#            rl_summary_writer = tf.summary.FileWriter(FLAGS.output_dir + 'summaries/rl')

#            f1_list = []
#            heq_list = []
#            dheq_list = []
#            yesno_list, followup_list = [], []

# Training cycle
#            for step, batch in enumerate(train_batches):
#                if step > num_train_steps:
# this means the learning rate has been decayed to 0
#                    break

# time1 = time()
#                batch_features, batch_slice_mask, batch_slice_num, output_features = batch

#                fd = convert_features_to_feed_dict(batch_features) # feed_dict
#                fd_output = convert_features_to_feed_dict(output_features)

#                if FLAGS.better_hae:
#                    turn_features = get_turn_features(fd['metadata'])
#                    fd['history_answer_marker'] = fix_history_answer_marker_for_bhae(fd['history_answer_marker'], turn_features)

#                if FLAGS.history_ngram != 1:
#                    batch_slice_mask, group_batch_features = group_histories(batch_features, fd['history_answer_marker'],
#                                                                                    batch_slice_mask, batch_slice_num)
#                    fd = convert_features_to_feed_dict(group_batch_features)


#                try:
#                    _, train_summary, total_loss_res = sess.run([train_op, merged_summary_op, total_loss],
#                                                    feed_dict={unique_ids: fd['unique_ids'], input_ids: fd['input_ids'],
#                                                    input_mask: fd['input_mask'], segment_ids: fd['segment_ids'],
#                                                    start_positions: fd_output['start_positions'], end_positions: fd_output['end_positions'],
#                                                    history_answer_marker: fd['history_answer_marker'], slice_mask: batch_slice_mask,
#                                                    slice_num: batch_slice_num,
#                                                    aux_start_positions: fd['start_positions'], aux_end_positions: fd['end_positions'],
#                                                    yesno_labels: fd_output['yesno'], followup_labels: fd_output['followup'], training: True})
#                except Exception as e:
#                    print('training, features length: ', len(batch_features))
#                    print(e)
#                    traceback.print_tb(e.__traceback__)

#                train_summary_writer.add_summary(train_summary, step)
#                train_summary_writer.flush()
# print('attention weights', attention_weights_res)
#                print('training step: {}, total_loss: {}'.format(step, total_loss_res))
# time2 = time()
# print('train step', time2-time1)


# if (step % 3000 == 0 or                 (step >= FLAGS.evaluate_after and step % FLAGS.evaluation_steps == 0)) and                 step != 0:
#                if step >= FLAGS.evaluate_after and step % FLAGS.evaluation_steps == 0:

#                    val_total_loss = []
#                    all_results = []
#                    all_output_features = []

#                    val_batches = cqa_gen_example_aware_batches_v2(val_features, val_example_tracker, val_variation_tracker,
#                                                    val_example_features_nums, FLAGS.predict_batch_size, 1, shuffle=False)

#                    for val_batch in val_batches:
# time3 = time()
#                        batch_results = []
#                        batch_features, batch_slice_mask, batch_slice_num, output_features = val_batch

#                        try:
#                            all_output_features.extend(output_features)

#                            fd = convert_features_to_feed_dict(batch_features) # feed_dict
#                            fd_output = convert_features_to_feed_dict(output_features)

#                            if FLAGS.better_hae:
#                                turn_features = get_turn_features(fd['metadata'])
#                                fd['history_answer_marker'] = fix_history_answer_marker_for_bhae(fd['history_answer_marker'], turn_features)

#                            if FLAGS.history_ngram != 1:
#                                batch_slice_mask, group_batch_features = group_histories(batch_features, fd['history_answer_marker'],
#                                                                                    batch_slice_mask, batch_slice_num)
#                                fd = convert_features_to_feed_dict(group_batch_features)

#                            start_logits_res, end_logits_res,                         yesno_logits_res, followup_logits_res,                         batch_total_loss,                         attention_weights_res = sess.run([start_logits, end_logits, yesno_logits, followup_logits,
#                                                            total_loss, attention_weights],
#                                        feed_dict={unique_ids: fd['unique_ids'], input_ids: fd['input_ids'],
#                                        input_mask: fd['input_mask'], segment_ids: fd['segment_ids'],
#                                        start_positions: fd_output['start_positions'], end_positions: fd_output['end_positions'],
#                                        history_answer_marker: fd['history_answer_marker'], slice_mask: batch_slice_mask,
#                                        slice_num: batch_slice_num,
#                                        aux_start_positions: fd['start_positions'], aux_end_positions: fd['end_positions'],
#                                        yesno_labels: fd_output['yesno'], followup_labels: fd_output['followup'], training: False})

#                            val_total_loss.append(batch_total_loss)

#                            key = (tuple([val_examples[f.example_index].qas_id for f in output_features]), step)
#                            attention_dict[key] = {'batch_slice_mask': batch_slice_mask, 'attention_weights_res': attention_weights_res,
#                                                'batch_slice_num': batch_slice_num, 'len_batch_features': len(batch_features),
#                                                'len_output_features': len(output_features)}

#                            for each_unique_id, each_start_logits, each_end_logits, each_yesno_logits, each_followup_logits                                 in zip(fd_output['unique_ids'], start_logits_res, end_logits_res, yesno_logits_res, followup_logits_res):
#                                each_unique_id = int(each_unique_id)
#                                each_start_logits = [float(x) for x in each_start_logits.flat]
#                                each_end_logits = [float(x) for x in each_end_logits.flat]
#                                each_yesno_logits = [float(x) for x in each_yesno_logits.flat]
#                                each_followup_logits = [float(x) for x in each_followup_logits.flat]
#                                batch_results.append(RawResult(unique_id=each_unique_id, start_logits=each_start_logits,
#                                                            end_logits=each_end_logits, yesno_logits=each_yesno_logits,
#                                                            followup_logits=each_followup_logits))

#                            all_results.extend(batch_results)
#                        except Exception as e:
#                            print('batch dropped because too large!')
#                            print('validating, features length: ', len(batch_features))
#                            print(e)
#                            traceback.print_tb(e.__traceback__)
# time4 = time()
# print('val step', time4-time3)
#                    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions_{}.json".format(step))
#                    output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions_{}.json".format(step))
#                    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "output_null_log_odds_file_{}.json".format(step))

# time5 = time()
#                    write_predictions(val_examples, all_output_features, all_results,
#                                    FLAGS.n_best_size, FLAGS.max_answer_length,
#                                    FLAGS.do_lower_case, output_prediction_file,
#                                    output_nbest_file, output_null_log_odds_file)
# time6 = time()
# print('write all val predictions', time6-time5)
#                    val_total_loss_value = np.average(val_total_loss)


# call the official evaluation script
#                    val_summary = tf.Summary()
# time7 = time()
#                    val_eval_res = external_call(val_file_json, output_prediction_file)
# time8 = time()
# print('external call', time8-time7)
#                    val_f1 = val_eval_res['f1']
#                    val_followup = val_eval_res['followup']
#                    val_yesno = val_eval_res['yes/no']
#                    val_heq = val_eval_res['HEQ']
#                    val_dheq = val_eval_res['DHEQ']

#                    heq_list.append(val_heq)
#                    dheq_list.append(val_dheq)
#                    yesno_list.append(val_yesno)
#                    followup_list.append(val_followup)

#                    val_summary.value.add(tag="followup", simple_value=val_followup)
#                    val_summary.value.add(tag="val_yesno", simple_value=val_yesno)
#                    val_summary.value.add(tag="val_heq", simple_value=val_heq)
#                    val_summary.value.add(tag="val_dheq", simple_value=val_dheq)

#                    print('evaluation: {}, total_loss: {}, f1: {}, followup: {}, yesno: {}, heq: {}, dheq: {}\n'.format(
#                        step, val_total_loss_value, val_f1, val_followup, val_yesno, val_heq, val_dheq))
#                    with open(FLAGS.output_dir + 'step_result.txt', 'a') as fout:
#                            fout.write('{},{},{},{},{},{},{}\n'.format(step, val_f1, val_heq, val_dheq,
#                                                                    val_yesno, val_followup, FLAGS.output_dir))

#                    val_summary.value.add(tag="total_loss", simple_value=val_total_loss_value)
#                    val_summary.value.add(tag="f1", simple_value=val_f1)
#                    f1_list.append(val_f1)
#                    val_summary_writer.add_summary(val_summary, step)
#                    val_summary_writer.flush()

#                    save_path = saver.save(sess, '{}/model_{}.ckpt'.format(FLAGS.output_dir, step))
#                    print('Model saved in path', save_path)


# In[4]:


#    best_f1 = max(f1_list)
#    best_f1_idx = f1_list.index(best_f1)
#    best_heq = heq_list[best_f1_idx]
#    best_dheq = dheq_list[best_f1_idx]
#    best_yesno = yesno_list[best_f1_idx]
#    best_followup = followup_list[best_f1_idx]
#    with open(FLAGS.output_dir + 'result.txt', 'w') as fout:
#        fout.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(best_f1, best_heq, best_dheq, best_yesno, best_followup,
#                                                    FLAGS.MTL_lambda, FLAGS.MTL_mu, FLAGS.MTL, FLAGS.mtl_input,
#                                                    FLAGS.history_attention_input, FLAGS.output_dir))