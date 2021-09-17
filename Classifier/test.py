"""
For Creating Correctly classisifed Test Files
__author__= Sohaib Kiani
"""
import os
import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf
from datetime import datetime
import pickle
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='', help='Location for the dataset')
parser.add_argument('-d', '--data_set', type=str, default='imagenet', help='Can be either cifar|imagenet|gtsrb')
# model
parser.add_argument('-m', '--model_dir', type=str, default='ResNet50', help='Classifier Model')
parser.add_argument('-l', '--num_class', type=int, default=10, help='Number of Output classes')
# optimization
parser.add_argument('-b', '--batch_size', type=int, default=200, help='Batch size during training per GPU')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
# -----------------------------------------------------------------------------
# fix random seed for reproducibility
np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)
# initialize data loaders for train/test splits
args.data_dir = os.path.join( os.getcwd(), '..', f'data/{args.data_set}')
args.model_dir = os.path.join( os.getcwd(), '..', f'Classifier/{args.data_set}_Model/ResNet50_ckpt' )
if args.data_set == 'imagenet':
    import data.imagenet_data as imagenet_data
    data = imagenet_data.DataLoader(args.data_dir)
    args.num_class = 16
elif args.data_set == 'gtsrb':
    import data.GTSRB_data as gtsrb_data
    data = gtsrb_data.DataLoader(args.data_dir)
    args.num_class = 43
elif args.data_set == 'cifar':
    import data.cifar10_data as cifar_data
    data = cifar_data.CIFAR10Data(args.data_dir)
    args.num_class = 10
else:
    raise("unsupported dataset")
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

from ResNet import Model
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='eval',num_class = args.num_class)
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,tf.train.latest_checkpoint(args.model_dir))
    print ("Test Sample",data.eval_data.n)
    # Main training loop
    iterations = int(data.eval_data.n / args.batch_size)
    correct_pred = 0
    samples = []
    label = []
    m_label = []
    for ii in range(iterations):
      x_batch, y_batch = data.eval_data.get_next_batch(args.batch_size,
                                                         multiple_passes=False)
      nat_dict = {model.x_input: x_batch,
                  model.y_input: y_batch}
      count_correct,y_pred = sess.run([model.num_correct,model.predictions], feed_dict=nat_dict)
      ind = np.where(y_pred == y_batch)
      samples.extend(x_batch[ind])
      label.extend(y_batch[ind])
      m_label.extend(y_pred[ind])
      correct_pred += count_correct
    print('Accuracy {:.4}%'.format(correct_pred/data.eval_data.n * 100))
    print (np.shape(samples))
    print (label[0:10])
    print (m_label[0:10])
    print ("Range of data should be 0 - 255 and actual is: ",str(np.min(samples))+" "+str(np.max(samples)))
    if np.max(samples)  <= 1.0:
        samples= samples * 255.0
    data = {'x':np.array(samples),'y':np.array(label)}
    pickle.dump(data, open(f"{args.data_dir}/test_c", 'wb'))
