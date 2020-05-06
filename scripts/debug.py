from pt_cqa_model import FineGrainedHistoryAttentionNet
from argparse import ArgumentParser
import torch

if __name__ == '__main__':

    parser = ArgumentParser(
        description='QA model')
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size for training and predicting")
    parser.add_argument("--max_seq_length", default=12, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--bert_hidden", default=4, type=int, help="bert hidden units, 768 or 1024")                                                                        
    parser.add_argument("--max_considered_history_turns", default=11, type=int, help="we only consider k history turns that immediately proceed the current turn, when generating preprocessed features,")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    net = FineGrainedHistoryAttentionNet(args)

    bert_representation = torch.ones([8, 12, 4])
    mtl_input = torch.ones([8, 4])
    slice_mask = [3, 3, 2, 1, 1, 1, 1, 1]
    slice_num = 3

    new_bert_representation, new_mtl_input, probs = net.forward(bert_representation, mtl_input, slice_mask, slice_num)

#    print(new_bert_representation.shape)
#    print(new_mtl_input.shape)
#    print(probs.shape)