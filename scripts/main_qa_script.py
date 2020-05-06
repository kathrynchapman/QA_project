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
from time import sleep

import numpy as np
from copy import deepcopy
import pickle
import itertools
import time
# from time import time
import traceback
import datetime
from os import listdir
from os.path import isfile, join, isdir

from tqdm import tqdm, trange
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import modeling_bert, BertConfig, BertTokenizer#, get_linear_schedule_with_warmup
from pt_cqa_supports import *
# from cqa_flags import FLAGS
from pt_cqa_model import *
from pt_cqa_gen_batches import cqa_gen_example_aware_batches_v2
# from cqa_rl_supports import *
from scorer import external_call  # quac official evaluation script
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
    

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

logger = logging.getLogger(__name__)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def load_data(file, tokenizer):
    """
    Read in the training data and generate examples, features and batches
    :param file: str path to Qauc data file
    :param tokenizer: BERT tokenizer to apply to the textual data
    :return:
    """
    examples = None
    num_train_steps = None
    num_warmup_steps = None
    train_or_dev = 'train' if 'train' in file else 'dev'
    full_or_mini = 'full' if not args.load_small_portion else 'mini'

    if args.load_small_portion:
        examples = read_quac_examples(input_file=file, is_training=True)[:300]
    else:
        examples = read_quac_examples(input_file=file, is_training=True)

    features_fname = args.cache_dir + '/{}_features_{}_{}.pkl'.format(
        train_or_dev, full_or_mini, args.max_considered_history_turns)
    example_tracker_fname = args.cache_dir + '/{}_example_tracker_{}_{}.pkl'.format(
        train_or_dev, full_or_mini, args.max_considered_history_turns)
    variation_tracker_fname = args.cache_dir + '/{}_variation_tracker_{}_{}.pkl'.format(
        train_or_dev, full_or_mini, args.max_considered_history_turns)
    example_features_nums_fname = args.cache_dir + '/{}_example_features_nums_{}_{}.pkl'.format(
        train_or_dev, full_or_mini, args.max_considered_history_turns)
    try:
        print('attempting to load {} features from cache'.format(train_or_dev))
        with open(features_fname, 'rb') as handle:
            features = pickle.load(handle)
        with open(example_tracker_fname, 'rb') as handle:
            example_tracker = pickle.load(handle)
        with open(variation_tracker_fname, 'rb') as handle:
            variation_tracker = pickle.load(handle)
        with open(example_features_nums_fname, 'rb') as handle:
            example_features_nums = pickle.load(handle)
    except:
        print('{} feature cache does not exist, generating'.format(train_or_dev))
        features, example_tracker, variation_tracker, example_features_nums = convert_examples_to_variations_and_then_features(
            examples=examples, tokenizer=tokenizer,
            max_seq_length=args.max_seq_length, doc_stride=args.doc_stride,
            max_query_length=64,
            max_considered_history_turns=args.max_considered_history_turns,
            is_training=train_or_dev=='train')
        with open(features_fname, 'wb') as handle:
            pickle.dump(features, handle)
        with open(example_tracker_fname, 'wb') as handle:
            pickle.dump(example_tracker, handle)
        with open(variation_tracker_fname, 'wb') as handle:
            pickle.dump(variation_tracker, handle)
        with open(example_features_nums_fname, 'wb') as handle:
            pickle.dump(example_features_nums, handle)
        print('{} features generated'.format(train_or_dev))

    temp_batches = cqa_gen_example_aware_batches_v2(features, example_tracker, variation_tracker,
                                                          example_features_nums,
                                                          batch_size=args.batch_size, num_epoches=1, shuffle=True)
    num_batches = len(list(temp_batches))


    return features, example_tracker, variation_tracker, example_features_nums, num_batches, examples

def train(train_file, tokenizer):
    train_features, train_example_tracker, train_variation_tracker, train_example_features_nums, \
    train_num_batches, train_examples = load_data(train_file, tokenizer)

    print("***** Running training *****")
    print("  Num orig examples = ", len(train_examples))
    print("  Num train_features = ", len(train_features))
    print("  Num train batches = ", train_num_batches)
    print("  Batch size = ", args.batch_size)
    print("  Num steps = ", args.num_train_steps)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    if args.num_train_steps > 0:
        t_total = args.num_train_steps
        args.num_epochs = args.num_train_steps // train_num_batches + 1

    set_seed(args)

    model = MTLModel(args)

    # model.to(device)
    model.zero_grad()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.num_train_steps
    )
    loss_fnct = MTLLoss(args)
    loss_log = tqdm(total=0, position=2, bar_format='{desc}')
    lr_log = tqdm(total=0, position=3, bar_format='{desc}')
    losses = []

    train_iterator = trange(
        epochs_trained, int(args.num_epochs), desc="Epoch", disable=False,
    )
    train_batches = cqa_gen_example_aware_batches_v2(train_features, train_example_tracker, train_variation_tracker,
                                                     train_example_features_nums,
                                                     batch_size=args.batch_size, num_epoches=1, shuffle=True)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_batches, desc="Iteration", disable=False, total=train_num_batches)
        for step, batch in enumerate(epoch_iterator):
            steps_trained_in_current_epoch += 1
            batch_features, batch_slice_mask, batch_slice_num, output_features = batch
            model.train()

            fd = convert_features_to_feed_dict(args, batch_features)  # feed_dict

            fd_output = convert_features_to_feed_dict(args, output_features)

            if args.do_MTL:
                (start_logits, end_logits), yesno_logits, followup_logits, attention_weights = model(fd,
                                                                                  batch_slice_mask, batch_slice_num)
                total_loss = loss_fnct.compute_total_loss(fd_output, start_logits, end_logits,
                                                          yesno_logits, followup_logits)
            else:
                start_logits, end_logits, attention_weights = model(fd, batch_slice_mask, batch_slice_num)
                total_loss = loss_fnct.compute_total_loss(fd_output, start_logits, end_logits)

            logging_loss = total_loss.item()
            losses.append(logging_loss)

            if step % 5 == 0:
                learning_rate_scalar = scheduler.get_last_lr()[0]
                loss_log.set_description_str(f'Current loss: {logging_loss}')
            lr_log.set_description_str(f'Current learning rate: {learning_rate_scalar}')
            epoch_iterator.update(1)
            optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            if global_step > args.num_train_steps:
                output_dir = os.path.join(args.output_dir + 'saved_checkpoints/checkpoint-{}/'.format(global_step))
                # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                torch.save(model.state_dict(), output_dir + "state_dict.pt")
                avg_loss = np.mean(losses)
                runtime_mins = (time.time() - start_time) / 60
                with open(args.output_dir + '/summaries/train/' + 'training_summary.txt', 'w') as f:
                    f.write("Summary of training executed on " + date_and_time.strftime('%d') + ' ' +
                            date_and_time.strftime("%B") + ' ' + date_and_time.strftime('%Y') + ' at ' +
                            date_and_time.strftime('%H') + ':' + date_and_time.strftime('%M') + '\n')
                    f.write("Final loss: " + str(total_loss.item()) + '\n')
                    f.write("Average loss: " + str(avg_loss) + '\n')
                    f.write("Number of training steps: " + str(global_step) + '\n')
                    f.write("Final learning rate: " + str(learning_rate_scalar) + '\n')
                    f.write("Training time: " + str(round(runtime_mins)) + ' minutes' + '\n')

                break

            # to reference for saving model
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir + 'saved_checkpoints/checkpoint-{}/'.format(global_step))
                # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                torch.save(model.state_dict(), output_dir + "state_dict.pt")
                # https://pytorch.org/tutorials/beginner/saving_loading_models.html

                #
                # model_to_save.save_pretrained(output_dir)
                # tokenizer.save_pretrained(output_dir)
                #
                # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                # logger.info("Saving model checkpoint to %s", output_dir)
                #
                # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                # logger.info("Saving optimizer and scheduler states to %s", output_dir)
        steps_trained_in_current_epoch = 0
        train_batches = cqa_gen_example_aware_batches_v2(train_features, train_example_tracker,
                                                         train_variation_tracker, train_example_features_nums,
                                                         batch_size=args.batch_size, num_epoches=1, shuffle=True)



attention_dict = {}
RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits", "yesno_logits", "followup_logits"])


def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


def evaluate(dev_file, tokenizer):
    val_summary_writer = SummaryWriter()
    val_total_loss = []
    all_results = []
    all_output_features = []

    f1_list = []
    heq_list = []
    dheq_list = []
    yesno_list, followup_list = [], []

    if args.eval_checkpoint:
        checkpoints_to_evaluate = ['checkpoint-' + args.eval_checkpoint]
    elif args.eval_all_checkpoints:
        checkpoints_to_evaluate = listdir(args.output_dir + 'saved_checkpoints/')
    else:
        # choose the checkpoint directory ending in the highest number (i.e. the last saved checkpoint)
        checkpoints_to_evaluate = ['checkpoint-' + str(
            max([int(f.split('-')[1]) for f in listdir(args.output_dir + 'saved_checkpoints/') if
                 isdir(join(args.output_dir + 'saved_checkpoints/', f))]))]

    for checkpoint in checkpoints_to_evaluate:

        state_dict_path = '{}saved_checkpoints/{}/state_dict.pt'.format(args.output_dir,
                                                                        checkpoint)

        model = MTLModel(args)

        model.load_state_dict(torch.load(state_dict_path))

        dev_features, dev_example_tracker, dev_variation_tracker, dev_example_features_nums, \
        dev_num_batches, dev_examples = load_data(dev_file, tokenizer)

        print("***** Running evaluation *****")
        print("  Num orig examples = ", len(dev_examples))
        print("  Num dev_features = ", len(dev_features))
        print("  Num dev batches = ", dev_num_batches)
        print("  Batch size = ", args.batch_size)

        set_seed(args)


        dev_batches = cqa_gen_example_aware_batches_v2(dev_features, dev_example_tracker, dev_variation_tracker,
                                                         dev_example_features_nums,
                                                         batch_size=args.batch_size, num_epoches=1, shuffle=False)

        dev_iterator = tqdm(dev_batches, desc="Iteration", disable=False, total=dev_num_batches)
        for step, batch in enumerate(dev_iterator):
            model.eval()
            batch_results = []
            batch_features, batch_slice_mask, batch_slice_num, output_features = batch


            all_output_features.extend(output_features)

            fd = convert_features_to_feed_dict(args, batch_features)  # feed_dict

            fd_output = convert_features_to_feed_dict(args, output_features)


            with torch.no_grad():
                inputs = {
                    "fd": fd,
                    "batch_slice_mask": batch_slice_mask,
                    "batch_slice_num": batch_slice_num,
                }

                if args.do_MTL:
                    (start_logits, end_logits), yesno_logits, followup_logits, attention_weights = model(**inputs)
                else:
                    start_logits, end_logits, attention_weights = model(**inputs)



            key = (tuple([dev_examples[f.example_index].qas_id for f in output_features]), step)
            attention_dict[key] = {'batch_slice_mask': batch_slice_mask, 'attention_weights_res': attention_weights,
                                   'batch_slice_num': batch_slice_num, 'len_batch_features': len(batch_features),
                                   'len_output_features': len(output_features)}

            for each_unique_id, each_start_logits, each_end_logits, each_yesno_logits, each_followup_logits \
                    in zip(fd_output['unique_ids'], start_logits, end_logits, yesno_logits,
                           followup_logits):
                each_unique_id = int(each_unique_id)
                each_start_logits = [float(x) for x in each_start_logits.tolist()]
                each_end_logits = [float(x) for x in each_end_logits.tolist()]
                each_yesno_logits = [float(x) for x in each_yesno_logits.tolist()]
                each_followup_logits = [float(x) for x in each_followup_logits.tolist()]
                batch_results.append(RawResult(unique_id=each_unique_id, start_logits=each_start_logits,
                                               end_logits=each_end_logits, yesno_logits=each_yesno_logits,
                                               followup_logits=each_followup_logits))

            all_results.extend(batch_results)


        output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(step))
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(step))
        output_null_log_odds_file = os.path.join(args.output_dir, "output_null_log_odds_file_{}.json".format(step))


        write_predictions(dev_examples, all_output_features, all_results,
                          args.n_best_size, args.max_answer_length,
                          args.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file)

        # -----------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------

        # time6 = time()
        # print('write all val predictions', time6-time5)
        val_total_loss_value = np.average(val_total_loss)

        # call the official evaluation script
        # val_summary = tf.Summary()
        # time7 = time()
        val_file_json = json.load(open(dev_file, 'r'))['data']
        val_eval_res = external_call(val_file_json, output_prediction_file)
        # time8 = time()
        # print('external call', time8-time7)
        val_f1 = val_eval_res['f1']
        val_followup = val_eval_res['followup']
        val_yesno = val_eval_res['yes/no']
        val_heq = val_eval_res['HEQ']
        val_dheq = val_eval_res['DHEQ']

        heq_list.append(val_heq)
        dheq_list.append(val_dheq)
        yesno_list.append(val_yesno)
        followup_list.append(val_followup)

        # val_summary.value.add(tag="followup", simple_value=val_followup)
        # val_summary.value.add(tag="val_yesno", simple_value=val_yesno)
        # val_summary.value.add(tag="val_heq", simple_value=val_heq)
        # val_summary.value.add(tag="val_dheq", simple_value=val_dheq)

        print('evaluation: {}, total_loss: {}, f1: {}, followup: {}, yesno: {}, heq: {}, dheq: {}\n'.format(
            step, val_total_loss_value, val_f1, val_followup, val_yesno, val_heq, val_dheq))
        with open(args.output_dir + 'step_result.txt', 'a') as fout:
            fout.write('{},{},{},{},{},{},{}\n'.format(step, val_f1, val_heq, val_dheq,
                                                       val_yesno, val_followup, args.output_dir))

        # val_summary.value.add(tag="total_loss", simple_value=val_total_loss_value)
        # val_summary.value.add(tag="f1", simple_value=val_f1)
        f1_list.append(val_f1)
        # val_summary_writer.add_summary(val_summary, step)
        # val_summary_writer.flush()

        # save_path = saver.save(sess, '{}/model_{}.ckpt'.format(FLAGS.output_dir, step))
        # print('Model saved in path', save_path)


        # -----------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------






if __name__ == '__main__':
    date_and_time = datetime.datetime.now()
    start_time = time.time()

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
    parser.add_argument('--do_eval', action='store_true', help='Whether to evaluate on dev set.')
    parser.add_argument('--do_predict', action="store_true", help='Whether to predict or not.')
    parser.add_argument('--eval_checkpoint', default=None, type=str, help='Specific checkpoint number to evaluate; default is latest checkpoint.')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--load_small_portion", action="store_true", help="Load a small portion of data during dev.")
    parser.add_argument("--max_seq_length", default=384, type=int, help="The maximum total input sequence "
                                                                        "length after tokenization.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--doc_stride", default=128, type=int, help="When splitting up a long document into chunks,"
                                                                    "how much stride to take between chunks.")
    parser.add_argument("--max_considered_history_turns", default=11, type=int, help="we only consider k history turns "
                                                                                     "that immediately proceed the current turn, when generating preprocessed features,")
    parser.add_argument("--warmup_proportion", default=0.1, help="Proportion of training to perform linear "
                                                                 "learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument("--history_attention_input", default='reduce_mean', type=str,
                        help="CLS, reduce_mean, reduce_max")
    parser.add_argument("--mtl_input", default="reduce_mean", type=str, help="CLS, reduce_mean, reduce_max")
    parser.add_argument("--aux_shared", default=False, type=bool,
                        help="wheter to share the aux prediction layer with the main convqa model")
    parser.add_argument("--disable_attention", action="store_true", help="disable the history attention module")
    parser.add_argument("--fine_grained_attention", action="store_true", help="Use fine-grained history attention module")
    parser.add_argument("--batch_size", default=24, type=int, help="Batch size for training and predicting")
    parser.add_argument("--num_epochs", default=0, type=int, help="Number of training epochs")
    parser.add_argument("--do_MTL", default=True, type=bool, help="Whether to do multi-task learning")
    parser.add_argument("--MTL_lambda", default=0.1, type=float, help="total loss = (1 - 2 * lambda) * convqa_loss + "
                                                                      "lambda * followup_loss + lambda * yesno_loss")
    parser.add_argument("--MTL_mu", default=0.8, type=float, help="total loss = mu * convqa_loss + lambda * "
                                                                  "followup_loss + lambda * yesno_loss")
    parser.add_argument("--bert_hidden", default=768, type=int, help="bert hidden units, 768 or 1024")
    parser.add_argument("--num_train_steps", default=30000, type=int, help= "loss: the loss gap on reward set, f1: the f1 on reward set")
    parser.add_argument("--bert_version", default='bert-base-uncased', type=str, help="Which BERT model to use: bert-case-cased, bert-base-uncased, bert-large-cased, bert-large-uncased")
    parser.add_argument("--n_best_size", default=20, type=int, help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument('--max_answer_length', default=50, type=int, help="The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.")
    parser.add_argument("--do_lower_case", default=True, type=bool, help="Whether to lower case the input text. Should be True for uncased models and False for cased models.")
    parser.add_argument("--eval_all_checkpoints", action='store_true', help='Run eval script on all saved checkpoints. (Warning: will take a while)')
    args = parser.parse_args()
    args.output_dir = args.output_dir + '/' if args.output_dir[-1] != '/' else args.output_dir
    args.num_warmup_steps = int(args.num_train_steps * args.warmup_proportion)
    args.n_gpu = torch.cuda.device_count()

    if 'large' in args.bert_version:
        args.bert_hidden = 1024

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    logger.warning(
        "Device: %s, n_gpu: %s",
        device,
        args.n_gpu,
    )

    # set the seed for initialization
    set_seed(args)

    # get the BERT config using huggingface
    bert_config = BertConfig.from_pretrained(args.bert_version)
    args.bert_config = bert_config

    tb_writer = SummaryWriter()


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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir + '/summaries/train/')
        os.makedirs(args.output_dir + '/summaries/val/')
        os.makedirs(args.output_dir + '/summaries/rl/')

    tokenizer = BertTokenizer.from_pretrained(args.bert_version)

    dev_file = args.quac_data_dir + 'val_v0.2.json' if args.quac_data_dir[
                                                           -1] == '/' else args.quac_data_dir + '/val_v0.2.json'  # CHANGED TO VAL ONLY FOR TESTING
    train_file = args.quac_data_dir + 'train_v0.2.json' if args.quac_data_dir[
                                                               -1] == '/' else args.quac_data_dir + '/train_v0.2.json'


    if args.do_train:
        train(train_file, tokenizer)
    if args.do_eval:
        evaluate(dev_file, tokenizer)
