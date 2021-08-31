"""
To train ResNet Classifier
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

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='./data/CIFAR', help='Location for the dataset')
#parser.add_argument('-o', '--save_dir', type=str, default='CIFAR_Model', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet|gtsrb')
parser.add_argument('-t', '--save_interval', type=int, default=500, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--model', type=str, default='ResNet50', help='Classifier Model')
parser.add_argument('-c', '--num_class', type=int, default=10, help='Number of Output classes')
# optimization
parser.add_argument('-l', '--momentum', type=float, default=0.9, help='Base learning rate')
parser.add_argument('-e', '--weight_decay', type=float, default=0.0002, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=200, help='Batch size during training per GPU')
parser.add_argument('-p', '--dropout_p', type=float, default=0.7, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=20000, help='How many epochs to run in total?')
parser.add_argument('-ss', '--step_size_schedule', type=float, default=[[0, 0.01], [40000, 0.001], [60000, 0.0001]], help='trainin steps')
parser.add_argument('-g', '--num_output_steps', type=int, default=100, help='output display steps')

# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args.data_dir = os.path.join( os.getcwd(), '..', f'data/{args.data_set}' )
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/test splits

if args.data_set == 'imagenet':
    import data.imagenet_data as imagenet_data
    raw_imagenet = imagenet_data.DataLoader(args.data_dir)
    assert args.num_class == 16
elif args.data_set == 'gtsrb':
    import data.GTSRB_data as gtsrb_data
    raw_gtsrb = gtsrb_data.DataLoader(args.data_dir)
    assert args.num_class == 43
elif args.data_set == 'cifar':
    import data.cifar10_data as cifar_data
    raw_cifar = cifar_data.CIFAR10Data(args.data_dir)
    assert args.num_class == 10

else:
    raise("unsupported dataset")



from ResNet import Model
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train',num_class = args.num_class)

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in args.step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in args.step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent + args.weight_decay * model.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, args.momentum).minimize(
    total_loss,
    global_step=global_step)




# //////////// perform training //////////////
saver = tf.train.Saver(max_to_keep=2)

if not os.path.exists(f"{args.data_set}_Model/{args.model}_ckpt"):
    os.makedirs(f"{args.data_set}_Model/{args.model}_ckpt")


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if args.load_params == True:
        saver.restore(sess,tf.train.latest_checkpoint(f"{args.data_set}_Model/{args.model}_ckpt"))
    # initialize data augmentation
    if args.data_set == 'imagenet':
        data = imagenet_data.AugmentedImageNetData(raw_imagenet, sess, model)
    elif args.data_set == 'gtsrb':
        data = gtsrb_data.AugmentedGTSRBData(raw_gtsrb, sess, model)
    elif args.data_set == 'cifar':
        data = cifar_data.AugmentedCIFAR10Data(raw_cifar, sess, model)




    # Initialize the summary writer, global variables, and our time counter.



    # Main training loop
    for ii in range(args.max_epochs):

      # Output to stdout
          if ii % args.num_output_steps == 0:
                x_batch, y_batch = data.eval_data.get_next_batch(args.batch_size,
                                                                   multiple_passes=True)

                test_dict = {model.x_input: x_batch,
                            model.y_input: y_batch}

                nat_acc = sess.run(model.accuracy, feed_dict=test_dict)

                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100),flush = True)



          if ii % args.save_interval == 0:
            saver.save(sess,os.path.join(f"{args.data_set}_Model/{args.model}_ckpt", 'checkpoint'),global_step=global_step)

          # Actual training step
          x_batch, y_batch = data.train_data.get_next_batch(args.batch_size,
                                                             multiple_passes=True)

          train_dict = {model.x_input: x_batch,
                      model.y_input: y_batch}
          sess.run(train_step, feed_dict=train_dict)
