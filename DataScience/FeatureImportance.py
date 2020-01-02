#!/usr/bin/env python3
# ==========================================================================================
# Find feature importance
# ==========================================================================================
from DashboardMpi.helpers import vw, grid, sweep


'''
Usage example:
>python FeatureImportance.py -d data.json --ml_args "--cb_adf -l 0.01 --cb_type mtr" -n 5

Sample output:
=====================================
Testing a range of L1 regularization
L1: 1e-06 - Num of Features: 4197
L1: 1e-05 - Num of Features: 2741
L1: 1e-04 - Num of Features: 46
L1: 1e-03 - Num of Features: 13
=====================================

Inverting hashes of feature importance (l1 = 1e-03)
=====================================
Emotion0^contempt:194265:-0.0236295
Emotion0^disgust:251874:-0.0327635
Emotion0^fear:188970:1.0217
Emotion0^surprise:1982:-0.416561
Emotion1^anger:197213:-0.486541
Emotion1^contempt:94653:-0.34001
Emotion1^disgust:166742:-0.368214
Emotion1^fear:23093:-5.25565
Emotion2^anger:104155:-0.20074
Emotion2^contempt:104000:-0.457823
Emotion2^disgust:22454:-0.142982
Emotion2^fear:23283:-177.874
Emotion2^sadness:157471:-1.35735
=====================================
'''

import os, argparse, sys
from subprocess import check_output, DEVNULL

def get_pretty_feature(feature):
    tokens = feature.split('^')
    if tokens[0] == 'FromUrl':
        tokens[0] = 'Context'
    elif tokens[0] == 'i' or tokens[0] == 'j':
        tokens[0] = 'Action'
    return '.'.join(tokens)


def get_pretty_features(features):
    featurelist = features.split('*')
    pretty_feature_list = list(map(get_pretty_feature, featurelist))
    return " with ".join(pretty_feature_list)


def extract_features(fp, inv_hash):
    features = []
    with open(fp, 'r') as readable_model:
        for line in readable_model:
            if line.strip() == ':1':
                text = readable_model.read().split('\ncurrent_pass 1\n',1)[1].strip()
                break
            elif line.strip() == ':0':
                text = readable_model.read().split('\n:0\n',1)[1].strip()
                break
    if '\n' not in text:
        print ('no features found in model output file: {0}'.format(fp))
    else:
        for line in text.splitlines():
            data = line.split(':')
            if data[0] in inv_hash:
                features.append(inv_hash[data[0]])
            else:
                print ('missing hash value in inv_hash: {0}'.format(data[0]))
    return features

# read the invert hash file and return a dictionary that maps from hash value to feature.
def get_feature_inv_hash(invert_hash_list):
    inv_hash = {}
    for fp in invert_hash_list:
        text = open(fp).read().split('\n:0\n',1)[1].strip()
        if '\n' not in text:
            print ('no features found in invert has file: {0}.'.format(fp))
        else:
            for line in text.splitlines():
                data = line.split(':')
                if (len(data) == 3) and not data[1] in inv_hash:
                    inv_hash[data[1]] = data[0]
        return inv_hash

# return unique buckets of features from the feature funnel.
# sample: input => [['c','b','a','d','e'],['b','c','a'],['a']] returns output => [['a'], ['b', 'c'], ['d', 'e']]
def get_feature_buckets(features_funnel):
    union_features = []
    feature_buckets = []
    for features in reversed(features_funnel):
        unique_features = list(set(features) - set(union_features))
        if len(unique_features) > 0:
            unique_features.sort()
            feature_buckets.append(unique_features)
            union_features.extend(unique_features)
    return feature_buckets

def get_feature_importance(env, ml_args, min_num_features):
    if ' --l1 ' in ml_args:
        temp = ml_args.split(' --l1 ',1)
        ml_args = temp[0]
        if ' ' in temp[1]:
            temp = temp[1].split(' ',1)
            ml_args += ' ' + temp[1]
            l1 = float(temp[0])
        else:
            l1 = float(temp[1])
    else:
        l1 = 1e-7

    inv_hash = get_feature_inv_hash(env.invert_hash_provider.list())
    print('\n=====================================')

    print('Testing a range of L1 regularization')
    all_features_funnel = []
    fi_base = {'#base': ml_args + ' --dsjson --readable_model'}
    fi_grid = grid.generate_fi_grid(l1)
    fi_candidates = sweep.sweep(fi_grid, env, fi_base)

    for index, fp in enumerate(env.readable_model_provider.list()):
        l1 = float(fi_grid[0].points[index]['--l1'])
        features = extract_features(fp, inv_hash)
        num_features = len(features)
        print('L1: {0:.0e} - Num of Features: {1}, File - {2}'.format(l1, num_features, os.path.basename(fp)))
        all_features_funnel.append(features)

        # If we fall below the minimum number of features, then break out of the loop.
        if num_features < min_num_features:
            print('Number of features is {0} which is below the minimum of {1}. Exiting the loop with L1 value of: {2:.0e}'.format(num_features, min_num_features, l1))
            break

    print("feature funnel sizes: {0}".format([len(features) for features in all_features_funnel]))
    feature_buckets = get_feature_buckets(all_features_funnel)
    not_required_features = ['constant', 'action.constant']
    pretty_feature_buckets = [[get_pretty_features(feature) for feature in feature_bucket] for feature_bucket in feature_buckets]
    pretty_feature_buckets = [[f for f in bucket if f.lower() not in not_required_features] for bucket in pretty_feature_buckets]
    return [feature_buckets, pretty_feature_buckets]


def add_parser_args(parser):
    parser.add_argument('--ml_args', help="ML arguments (default: --cb_adf -l 0.01)", default='--cb_adf -l 0.01')
    parser.add_argument('-n', '--min_num_features', type=str, help="Minimum Number of features.", default='5')


def main(env, args):
    vw.check_vw_installed(env.logger)
    return get_feature_importance(env, args.ml_args, int(args.min_num_features))
