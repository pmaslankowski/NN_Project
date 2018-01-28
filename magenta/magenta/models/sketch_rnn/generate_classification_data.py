# Generating classification data
# Script generates dataset of h values and input classes

# Important: this script assumes that model class (in file model.py) were modified
# and saves h values too. 
# Maybe you have to modify model.py in your anaconda directory 




import numpy as np
import time
import random
import cPickle
import codecs
import collections
import os
import math
import json
import tensorflow as tf
import pdb
from six.moves import xrange

import svgwrite # conda install -c omnia svgwrite=1.1.6
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *



# INPUT DATA FOR THIS SCRIPT:
data_dir = 'datasets/broccopigs'
models_root_dir = ''
model_dir = 'models/broccopigs'
dataset_files_with_labels = [
    ('datasets/broccopigs/broccoli.npz', 'broccoli'),
    ('datasets/broccopigs/pigs.npz', 'pig')   
]
number_of_examples = 3000
output_file = 'test1'



class2id = {
    'broccoli' : 0,
    'pig' : 1,
    'airplane' : 2,
    'bus' : 3,
    'cactus' : 4,
    'cat' : 5
}

def custom_load_datasets(datasets):
    train_strokes = None
    valid_strokes = None
    test_strokes = None
    data_dict = {}
    
    for data_filepath, label in datasets:
        data = np.load(data_filepath)
        print 'Loaded {}/{}/{} from {} ({})'.format(
            len(data['train']), len(data['valid']), len(data['test']),
            data_filepath, label)
    
        data_dict[label] = {'train': data['train'], 'valid': data['valid'], 'test': data['test']}
    
        if train_strokes is None:
            train_strokes = data['train']
            valid_strokes = data['valid']
            test_strokes = data['test']
        else:
            train_strokes = np.concatenate((train_strokes, data['train']))
            valid_strokes = np.concatenate((valid_strokes, data['valid']))
            test_strokes = np.concatenate((test_strokes, data['test']))

    all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
    num_points = 0
    for stroke in all_strokes:
        num_points += len(stroke)
    avg_len = num_points / len(all_strokes)
    print 'Dataset combined: {} ({}/{}/{}), avg len {}'.format(
        len(all_strokes), len(train_strokes), len(valid_strokes),
        len(test_strokes), int(avg_len))

  # calculate the max strokes we need.
    max_seq_len = utils.get_max_len(all_strokes)

    print 'max_seq_len %i.', max_seq_len

    all_train_set = utils.DataLoader(
        train_strokes,
        100,
        max_seq_length=max_seq_len,
        random_scale_factor=0.15,
        augment_stroke_prob=0.10)

    normalizing_scale_factor = all_train_set.calculate_normalizing_scale_factor()
    print 'normalizing_scale_factor %4.4f.', normalizing_scale_factor
  
    result = {}
    for label, dataset in data_dict.iteritems():
        print 'data loader for {} created'.format(label)
        result[label] = { key : utils.DataLoader(
            data[key], 
            1, 
            max_seq_length=max_seq_len,
            random_scale_factor=0.15,
            augment_stroke_prob=0.10)
            for key in dataset }
    
    print 'normalising {} dataset...'.format(label)
    for key in result[label]:
        result[label][key].normalize(normalizing_scale_factor)

    return result


def encode(input_strokes, stroke_max_length = 250, draw=True):
  strokes = to_big_strokes(input_strokes, stroke_max_length).tolist()
  strokes.insert(0, [0, 0, 1, 0, 0])
  seq_len = [len(input_strokes)]
  if draw:
    draw_strokes(to_normal_strokes(np.array(strokes)))
  return sess.run([eval_model.batch_z, eval_model.h], feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})


def generate_training_data_from_loader(N, data_loader, label):
    n = min(N, data_loader.num_batches)
    hs = np.vstack(
        (np.array([encode(stroke_3, data_loader.max_seq_length, False)[1][0]
            for stroke_3 
            in data_loader.get_batch(i)[0]])
        for i in range(n)))
    labels = np.full((data_loader.batch_size * n), class2id[label])
    return hs, labels


def generate_training_data(output_file, datasets, N):
    datasets_dict = custom_load_datasets(datasets)
    print 'Datasets loaded'
    
    hs = {}
    labels = {}
    for label, datasets in datasets_dict.iteritems():
        for part_label, dataloader in datasets.iteritems():
            print 'generating {} data for class {}'.format(part_label, label)
            new_hs, new_labels = generate_training_data_from_loader(N, dataloader, label)
            if part_label not in hs:
                hs[part_label] = new_hs
                labels[part_label] = new_labels
            else:
                hs[part_label] = np.concatenate((hs[part_label], new_hs))
                labels[part_label] = np.concatenate((labels[part_label], new_labels))
                
    np.savez_compressed(output_file, train_hs=hs['train'], valid_hs=hs['valid'], test_hs=hs['test'],
                       train_labels=labels['train'], valid_labels=labels['valid'], test_labels=labels['test'])
    print 'Saved to {}'.format(output_file)

if __name__ == '__main__':
    print 'CONFIGURATION:'
    print 'data_dir={}'.format(data_dir)
    print 'model_dir={}'.format(model_dir)
    print 'number_of_examples={}'.format(number_of_examples)
    print 'output_file={}'.format(output_file)

    [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env(data_dir, model_dir)
    reset_graph()
    model = Model(hps_model)
    eval_model = Model(eval_hps_model, reuse=True)
    sample_model = Model(sample_hps_model, reuse=True)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    load_checkpoint(sess, model_dir)
    generate_training_data(output_file, dataset_files_with_labels, number_of_examples)
    