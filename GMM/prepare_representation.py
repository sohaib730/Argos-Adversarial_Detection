"""Prepare representation for train/test/adversarial
__author__ = Sohaib"""

import os
import sys
import argparse
sys.path.append(os.path.join( os.getcwd(), f'Classifier/' ))
from ResNet import Model

from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
version = sys.version_info

Latent_Vector_Length = 640
class CMadry_Graph():
      """  Importing and running isolated TF graph """
      def __init__(self, loc,c):
          # Create local graph and use it in the session
          self.graph = tf.Graph()
          self.sess = tf.Session(graph=self.graph)
          model_file = tf.train.latest_checkpoint(loc)
          with self.graph.as_default():


              self.model = Model(mode='eval',num_class = c)
              saver = tf.train.Saver()
              saver.restore(self.sess, model_file)


          tf.reset_default_graph()
      def run(self, dict_adv):
          """ Running the activation operation previously imported """

          return self.sess.run([self.model.pen_ultimate,self.model.num_correct,self.model.predictions],
                                            feed_dict=dict_adv)

      def get_features(self,dict):
          """ Running the activation operation previously imported """
          return self.sess.run([self.model.pen_ultimate,self.model.predictions,self.model.num_correct],feed_dict=dict)

def load_cifar(filename):
  with open(filename, 'rb') as fo:
      if version.major == 3:
          data_dict = pickle.load(fo, encoding='bytes')
      else:
          data_dict = pickle.load(fo)
      assert data_dict[b'data'].dtype == np.uint8
      image_data = data_dict[b'data']
      image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
      return image_data, np.array(data_dict[b'labels'])

def load_data(dataset,method):
    if dataset == 'cifar':
        if method == 'train':
            train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
            train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
            train_labels = np.zeros(50000, dtype='int32')
            for ii, fname in enumerate(train_filenames):
                cur_images, cur_labels = load_cifar(os.path.join(args.data_dir, fname))
                train_images[ii * 10000 : (ii+1) * 10000, ...] = cur_images
                train_labels[ii * 10000 : (ii+1) * 10000, ...] = cur_labels
                x = train_images
                y = train_labels
        elif method == 'test':
            eval_filename = 'test_batch'
            eval_images, eval_labels = load_cifar(
                os.path.join(args.data_dir, eval_filename))
            x = eval_images
            y = eval_labels
        else:
            data = pickle.load(open(f"{args.data_dir}/Adversarial/{method}", mode="rb"))
            x = data['data']
            y = data['tlabel']

    if dataset == 'gtsrb':
        if method == 'train':
            x = []
            y = []
            for i in range(1,4):
                data = pickle.load(open(f"{args.data_dir}/batch{i}", mode="rb"))
                x_temp = data['data']
                y_temp = data['label']
                if i == 1:
                    x = x_temp
                    y = y_temp
                else:
                    x = np.append(x,x_temp,axis=0)
                    y = np.append(y,y_temp,axis = 0)
        elif method == 'test':
            data = pickle.load(open(f"{args.data_dir}/test", mode="rb"))
            x = data['data']
            y = data['label']
        else:
            data = pickle.load(open(f"{args.data_dir}/Adversarial/{method}", mode="rb"))
            x = data['data']
            y = data['tlabel']

    if dataset == 'imagenet':
        if method == 'train':
            path = f'{args.data_dir}/batch_1'
            data = pickle.load(open(path, mode="rb"))
            x = data['x']
            y = data['y']
        elif method == 'test':
            path = f'{args.data_dir}/test'
            data = pickle.load(open(path, mode="rb"))
            print (data.keys())
            x = data['x']
            y = data['y']
        else:
            path = f'{args.data_dir}/Adversarial/{method}'
            data = pickle.load(open(path, mode="rb"))
            print (data.keys())
            x = data['data']
            y = data['tlabel']

    print ("Range of data should be 0-255 and actual is: ",str(np.min(x))+" "+str(np.max(x)))
    print (x.shape)
    return x,y



def get_features(x,y,Predictor):
    num_eval_examples = x.shape[0]
    print (num_eval_examples)
    batch_size = 200
    num_batches = int (num_eval_examples / batch_size)
    print ("Num of batches",num_batches)
    data = []
    label = []
    correct_count = 0
    for i in range(num_batches):
        start = i * batch_size
        end = (i+1) * batch_size
        dict_predictor = {Predictor.model.x_input: x[start:end], Predictor.model.y_input: y[start:end]}
        x_embed,y_pred, cc = Predictor.get_features(dict_predictor)
        if (args.method == 'clean'):
            ind = np.where (y[start:end] == y_pred)
        else:
            ind = np.where (y[start:end] != y_pred)
        data.extend(x_embed[ind])
        label.extend(y_pred[ind])
        correct_count += cc
    data = np.array(data)
    label = np.array (label)
    label = label.reshape((-1,))

    print ("For Clean only correctly classified examples are saved and for adversarial only misclassified")
    print ("data shape",data.shape)
    print ("label shape",label.shape)
    print ("Accuracy",correct_count / num_eval_examples)
    Rep = {'h':data,'label':label}
    return Rep

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-f', '--method', type=str, default='', help="Generate representation for all clean|adversarial data")
    parser.add_argument('-d', '--dataset', type=str, default='cifar', help='dataset cifar|imagenet|gtsrb')

    parser.add_argument('-i', '--data_dir', type=str, default='', help='Location for loading data data')
    parser.add_argument('-o', '--save_dir', type=str, default='', help='Location for the saving representation')
    parser.add_argument('-m', '--model_dir', type=str, default='', help='Classifier Model')

    args = parser.parse_args()

    args.data_dir = os.path.join( os.getcwd(), f'data/{args.dataset}' )
    args.save_dir = os.path.join( os.getcwd(), f'data/{args.dataset}/Representation' )
    if not os.path.exists(args.save_dir):
        print(f"Creating Output Directory {args.save_dir}")
        os.makedirs(args.save_dir)
    args.model_dir = os.path.join( os.getcwd(), f'Classifier/{args.dataset}_Model/ResNet50_ckpt' )

    if args.method == 'adversarial':
        method = ['pgd_E4','pgd_E8','pgd_E16','fgsm_E4','fgsm_E8','fgsm_E16','mim_E4','mim_E8','mim_E16','CW','deepFool']
    elif args.method == 'clean':
        method = ['train','test']
    else:
        print ("-f flag should be clean or adversarial")
        sys.exit()

    n_classes = 0
    if args.dataset == 'cifar':
        n_classes = 10
    elif args.dataset == 'imagenet':
        n_classes = 16
    else:
        n_classes = 43

    Predictor=CMadry_Graph(args.model_dir,n_classes)

    for m in method:
        print (f"Representation for {args.dataset} and file {m}")
        x,y = load_data(args.dataset,m)
        d = get_features(x,y,Predictor)

        pickle.dump(d, open(f"{args.save_dir}/{m}", "wb"))
