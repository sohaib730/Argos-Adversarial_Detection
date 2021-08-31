
import os
import sys
import json
import argparse
import time


# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/scratch/kiani/Projects/pixel-cnn/')

import numpy as np
import tensorflow as tf
import pickle
import argparse

from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
from utils import plotting
from numpy import linalg as LA


class PixelCNN():
    def __init__(self, args,num_class):
        obs_shape = (32,32,3)
        self.num_labels = num_class
        # energy distance or maximum likelihood?
        if args.energy_distance:
            loss_fun = nn.energy_distance
        else:
            print ("Discretized Mix logistic Loss")
            loss_fun = nn.discretized_mix_logistic_loss

        init_batch_size = 16
        x_init = tf.placeholder(tf.float32, shape=(init_batch_size,) + obs_shape)
        y_init = tf.placeholder(tf.int32, shape=(init_batch_size,))
        h_init = tf.one_hot(y_init, self.num_labels)

        self.xs = tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape)
        self.ys = tf.placeholder(tf.int32, shape=(args.batch_size,))
        hs = tf.one_hot(self.ys, self.num_labels)



        ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
        self.model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': args.energy_distance }
        model = tf.make_template('model', model_spec)


        init_pass = model(x_init, h_init, init=True, dropout_p=0.5, **self.model_opt)




        # keep track of moving average
        all_params = tf.trainable_variables()
        ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
        maintain_averages_op = tf.group(ema.apply(all_params))
        ema_params = [ema.average(p) for p in all_params]

        # test
        out = model(self.xs, hs, ema=ema, dropout_p=0., **self.model_opt)
        self.loss_gen_test=loss_fun(self.xs, out)


        # convert loss to bits/dim
        self.bits_per_dim_test = self.loss_gen_test/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
        # init & save
        #save_dir='/scratch/kiani/Projects/pixel-cnn/model'
        #self.graph = tf.Graph()
        self.sess = tf.Session()
        #ckpt_file = save_dir + '/params_' + args.data_set + '.ckpt'
        initializer = tf.global_variables_initializer()
        saver = tf.train.Saver()
        print('restoring generator parameters from', args.ckpt_file)
        saver.restore(self.sess, args.ckpt_file)

            # FROM SAVED COLLECTION:
    def likelihood(self,x_dict):
        return self.sess.run(self.bits_per_dim_test,feed_dict = x_dict)
    def get_out(self,x,y):
        hs = tf.one_hot(self.y, self.num_labels)
        out = model(x, hs, ema=ema, dropout_p=0., **self.model_opt)
        self.loss_gen_test=loss_fun(self.xs, out)
        # convert loss to bits/dim
        self.bits_per_dim_test = self.loss_gen_test/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
        return self.bits_per_dim_test
