from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

        
def cqa_gen_example_aware_batches_v2(features, example_tracker, variation_tracker, example_features_nums, batch_size, num_epoches, shuffle=False):
    # this is for history attention. suppose example 1 has 3 variations (e1.1, e1.2, e1.3), and each variation has two features
    # due to the sliding window approach. so example 1 has features (e1.1.1, e1.1.2, e1.2.1, e1.2.2, e1.3.1, e1.3.2)
    # we put (e1.1.1, e1.2.1, e1.3.1) in a batch, because we will compute history attention on them and get the weighted sum
    # as the representation for e1. We also include features from the same example or other example in this batch and provide a slide mask
    # to distinguish them. So the batch looks like (e1.1.1, e1.2.1, e1.3.1, e1.1.2, e1.2.2, e1.3.2, e2.1.1, e2.2.1), 
    # and the slice mask looks like (3, 3, 2), with each elements denote the number of features for each (xample_index, feature_index) combo

    prev_e_tracker, prev_v_tracker = None, None
    f_tracker = 0 # feature tracker, denotes the feature index for each variation
    features_dict = {}
    for feature, e_tracker, v_tracker in zip(features, example_tracker, variation_tracker):
        # get the f_tracker
        if e_tracker == prev_e_tracker and v_tracker == prev_v_tracker:
            f_tracker += 1
        else:
            f_tracker = 0
        prev_e_tracker, prev_v_tracker = e_tracker, v_tracker
        
        key = (e_tracker, f_tracker)
        if key not in features_dict:                    
            features_dict[key] = []
        features_dict[key].append(feature)
        
    feature_groups = list(features_dict.values())

    if shuffle:
        np.random.seed(0)
        np.random.shuffle(feature_groups)
#         idx = np.random.permutation(len(feature_groups))
#         feature_groups = np.asarray(feature_groups)[idx]
    
    for _ in range(int(num_epoches)):
        # we greedily select all the features that belong the next feature group, 
        # as long as the sum of example_features does not exceed FLAGS.train_batch_size
        batch_features = []
        batch_slice_mask = []
        batch_slice_num = None
        
        # after the weighted sum of history, we get a new representation for the example feature
        # this feature will be fed into the prediction layer
        # this feature share the input_ids, etc with the entire feature group
        # we use this feature to compute loss
        output_features = [] 
        
        for feature_group in feature_groups:
            len_feature_group = len(feature_group)
            if len(batch_features) + len_feature_group <= batch_size:
                batch_features.extend(feature_group)
                batch_slice_mask.append(len_feature_group)
                output_features.append(feature_group[0])
            else:
                batch_slice_num = len(batch_slice_mask)
                batch_slice_mask += [1] * (batch_size - len(batch_slice_mask))
                yield batch_features, batch_slice_mask, batch_slice_num, output_features
                
                batch_features = []
                batch_slice_mask = []
                batch_slice_num = None
                output_features = []
                
                batch_features.extend(feature_group)
                batch_slice_mask.append(len_feature_group)
                output_features.append(feature_group[0])
        
        if len(batch_features) > 0:
            batch_slice_num = len(batch_slice_mask)
            batch_slice_mask += [1] * (batch_size - len(batch_slice_mask))
            yield batch_features, batch_slice_mask, batch_slice_num, output_features

        print('epoch finished!', 'shuffle={}'.format(shuffle))