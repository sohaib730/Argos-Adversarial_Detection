"""
___Extract descroptors/features and saves labelled data in data/<dataset>/Final_Descriptors
___author__ = Sohaib kiani
"""

import os
import sys
import numpy as np
import pickle
from sklearn.metrics import classification_report
from sklearn import metrics
import tensorflow as tf
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import entropy
import argparse
import json
import time

sys.path.append('pixel-cnn/')
from Pixel_CNN import PixelCNN
sys.path.append('Classifier/')
from ResNet import Model


parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-d', '--dataset', type=str, default='imagenet', help='dataset cifar|imagenet|gtsrb')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_false', help='Condition generative model on labels?')
parser.add_argument('-ed', '--energy_distance', dest='energy_distance', action='store_true', help='use energy distance in place of likelihood')  ### action will happen when pass arg, otherwise default is false
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size during training per GPU')
parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=1, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('-ns', '--num_samples', type=int, default=1, help='How many batches of samples to output.')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
#Trained Models
parser.add_argument('-o', '--ckpt_file', type=str, default='', help='PixelCNN Model')

args = parser.parse_args()
args.data_dir = f'data/{args.dataset}/Generated'
args.ckpt_file = f"pixel-cnn/Model_{args.dataset}/params_{args.dataset}.ckpt"

args.save_dir = f'data/{args.dataset}/Final_Descriptors'
if not os.path.exists(args.save_dir):
    print(f"Creating Output Directory {args.save_dir}")
    os.makedirs(args.save_dir)
args.model_dir = f'Classifier/{args.dataset}_Model/ResNet50_ckpt'
args.GMM_dir = f'GMM/GMM_Models/{args.dataset}_GMM'
if args.dataset == 'imagenet':
    args.num_labels = 16
    args.den = -10000
if args.dataset == 'cifar':
    args.num_labels = 10
    args.den = -1000
if args.dataset == 'gtsrb':
    args.num_labels = 43
    args.den = 100


print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

Latent_Vector_Length = 640
class CMadry_Graph():
      """  Importing and running isolated TF graph """
      def __init__(self, loc, labels):
          # Create local graph and use it in the session
          self.graph = tf.Graph()
          self.sess = tf.Session(graph=self.graph)
          model_file = tf.train.latest_checkpoint(loc)
          with tf.device('/gpu:1'):
              with self.graph.as_default():
                  self.model = Model(mode='eval',num_class=labels)
                  saver = tf.train.Saver()
                  saver.restore(self.sess, model_file)


              tf.reset_default_graph()
      def run(self, dict_adv):
          """ Running the activation operation previously imported """

          return self.sess.run([self.model.pen_ultimate,self.model.predictions, self.model.post_softmax],
                                            feed_dict=dict_adv)

      def get_features(self,dict):
          """ Running the activation operation previously imported """
          return self.sess.run(self.model.pen_ultimate,feed_dict=dict)
def fractional_distance(p_vec, q_vec, fraction=2.0):
        """
        This method implements the fractional distance metric.
        :param p_vec: vector one
        :param q_vec: vector two
        :param fraction: the fractional distance value (power)
        :return: the fractional distance between vector one and two
        """
        # memoization is used to reduce unnecessary calculations ... makes a BIG difference
        memoize = False
        if memoize:
            key = self.get_key(p_vec, q_vec)
            x = memoization.get(key)
            if x is None:
                diff = p_vec - q_vec
                diff_fraction = diff**fraction
                return max(math.pow(np.sum(diff_fraction), 1/fraction), self.e)
            else:
                return x
        else:
            diff = np.abs(p_vec - q_vec)
            diff_fraction = diff**fraction
            return math.pow(np.sum(diff_fraction), 1/fraction)

def get_data(path):
    data = pickle.load(open(path , mode='rb'))
    data['x'] = ((data['x'] + 1) / 2.0)*255
    return data

if __name__ == "__main__":
    PCNN = PixelCNN(args,args.num_labels)
    Predictor = CMadry_Graph(args.model_dir,args.num_labels)
    GMM = pickle.load(open(args.GMM_dir , mode='rb'))

    clean_path = os.path.join(args.data_dir,'g_test_c')
    cdata = get_data(clean_path)
    print (cdata['x'].shape[0])
    num_samples = int (cdata['x'].shape[0])
    print ("Sample Size",num_samples/5)
    batch_size = 100 * 5
    c = 0
    df_c = pd.DataFrame(columns=('dist','Uncer','ph_0','Px', 'Label'))
    for i in range(0,num_samples,batch_size):
        start = i
        end = (i+1) * batch_size
        x = cdata['x'][start:end]
        dict = {Predictor.model.x_input: x}
        h,y_pred,u = Predictor.run(dict)
        for k in range(0,batch_size,5):
                x[k] = (x[k] - 127.5)/ 127.5
                dict_gen = {PCNN.xs:x[np.newaxis,k],PCNN.ys:y_pred[np.newaxis,k]}
                Px = PCNN.likelihood(dict_gen)
                e1 = entropy(u[k],u[k+1])
                e2 = entropy(u[k],u[k+2])
                e3 = entropy(u[k],u[k+3])
                e4 = entropy(u[k],u[k+4])
                e = e1 + e2 + e3 + e4
                d = distance.euclidean(h[k],h[k+4])
                sample = np.expand_dims(h[k],axis=0)
                ph_0 = GMM[y_pred[k]].score_samples(sample)/(args.den)
                df_c.loc[c] = [d,e,ph_0[0],Px,1]
                c+=1
    print (df_c.head())
    df_c.to_pickle(f"{args.save_dir}/g_test_c")
    print (f"Dataset {args.dataset} with attack method g_test_c and Save as g_test_c")

    adv_file = ['g_deepFool','g_CW','g_pgd_E4','g_pgd_E8','g_pgd_E16','g_mim_E4','g_mim_E8','g_mim_E16','g_fgsm_E4','g_fgsm_E8','g_fgsm_E16']
    for file in adv_file:
        adv_path = os.path.join(args.data_dir,file)
        adata = get_data(adv_path)
        print (cdata['x'].shape[0])
        num_samples = int (cdata['x'].shape[0])
        print ("Sample Size",num_samples/5)
        c = 0
        df_a = pd.DataFrame(columns=('dist','Uncer','ph_0','Px', 'Label'))
        for i in range(0,num_samples,batch_size):
            start_time = time.time()
            start = i
            end = (i+1) * batch_size
            x = adata['x'][start:end]
            dict = {Predictor.model.x_input: x}
            h,y_pred,u = Predictor.run(dict)
            for k in range(0,batch_size,5):
                    x[k] = (x[k] - 127.5)/ 127.5
                    dict_gen = {PCNN.xs:x[np.newaxis,k],PCNN.ys:y_pred[np.newaxis,k]}
                    Px = PCNN.likelihood(dict_gen)
                    e1 = entropy(u[k],u[k+1])
                    e2 = entropy(u[k],u[k+2])
                    e3 = entropy(u[k],u[k+3])
                    e4 = entropy(u[k],u[k+4])
                    e = e1 + e2 + e3 + e4
                    d = distance.euclidean(h[k],h[k+4])
                    sample = np.expand_dims(h[k],axis=0)
                    ph_0 = GMM[y_pred[k]].score_samples(sample)/(args.den)
                    df_a.loc[c] = [d,e,ph_0[0],Px,-1]
                    c+=1
            end_time = time.time()
            print('Time taken to generate %d images: %.2f seconds' %
                  (batch_size, end_time - start_time))

        print (df_a.head())
        df_a.to_pickle(f"{args.save_dir}/{file}")
        print (f"Dataset {args.dataset} with attack method {file} and Save as {file}")
