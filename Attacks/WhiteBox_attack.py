## __Author__ = Sohaib Kiani
# Data: 06/2021

import pickle
import numpy as np
import tensorflow as tf
import math
import os
import sys
sys.path.append(os.path.join( os.getcwd(), 'Classifier/' ))
from ResNet import Model

import tensorflow_probability as tfp
tfd = tfp.distributions
from scipy.special import logsumexp
sys.path.append(os.path.join( os.getcwd(), 'pixel-cnn/' ))
from Pixel_CNN import PixelCNN
import argparse
import json
version = sys.version_info

parser = argparse.ArgumentParser()
# data I/O

parser.add_argument('-o', '--ckpt_file', type=str, default='', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str, default='imagenet', help='Can be either cifar|imagenet/gtsrb')
parser.add_argument('-t', '--save_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
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
parser.add_argument('-ns', '--num_samples', type=int, default=-1, help='How many batches of samples to output.')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
args.data_dir = os.path.join( os.getcwd(), f'data/{args.dataset}' )
args.save_dir = os.path.join( os.getcwd(), f'data/{args.dataset}/Adversarial')
if not os.path.exists(args.save_dir):
    print(f"Creating Output Directory {args.save_dir}")
    os.makedirs(args.save_dir)
args.model_dir = os.path.join( os.getcwd(), f'Classifier/{args.dataset}_Model/ResNet50_ckpt' )
args.GMM_dir = os.path.join( os.getcwd(), f'GMM/GMM_Models/{args.dataset}_GMM' )
args.ckpt_file = os.path.join( os.getcwd(),f'pixel-cnn/Model_{args.dataset}/params_{args.dataset}.ckpt' )

if args.dataset == 'cifar':
    n_classes = 10
elif args.dataset == 'imagenet':
    n_classes = 16
else:
    n_classes = 43
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

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

def load_test_data():
    if args.dataset == 'cifar':
        eval_filename = 'test_batch'
        eval_images, eval_labels = load_cifar(
            os.path.join(args.data_dir, eval_filename))
        x = eval_images
        y = eval_labels

    if args.dataset == 'gtsrb':
        
        data = pickle.load(open(f"{args.data_dir}/test" , mode='rb'))
        print (data.keys())
        x = data['data']
        y = data['labels']
        print ("Range of data should be 0 - 255 and actual is: ",str(np.min(data['data']))+" "+str(np.max(data['data'])))

    if args.dataset == 'imagenet':
        
        data = pickle.load(open(f"{args.data_dir}/test" , mode='rb'))
        print (data.keys())
        x = data['x']
        y = data['y']


    print ("Range of data should be 0 - 255 and actual is: ",str(np.min(x))+" "+str(np.max(x)))

    return x,y

class CMadry_Graph():
      """  Importing and running isolated TF graph """
      def __init__(self, loc,GMM):
          # Create local graph and use it in the session
          self.graph = tf.Graph()
          self.sess = tf.Session(graph=self.graph)
          model_file = tf.train.latest_checkpoint(loc)
          with self.graph.as_default():
              self.model = Model(mode='eval',num_class = n_classes)
              saver = tf.train.Saver()

              self.ind = tf.compat.v1.placeholder(tf.int64, shape=None)
              weight = []
              cov = []
              mean = []
              for i in range(n_classes):
                  weight.append(GMM[i].weights_)
                  cov.append(GMM[i].covariances_)
                  mean.append(GMM[i].means_)
              weight = np.array(weight)
              cov = np.array(cov)
              mean = np.array(mean)
              #gmm0 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=GMM[i].weights_),components_distribution=tfd.MultivariateNormalFullCovariance(loc=GMM[i].means_,covariance_matrix=GMM[i].covariances_))
              #i = 1
              gmm1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=weight),components_distribution=tfd.MultivariateNormalFullCovariance(loc=mean,covariance_matrix=cov))
              self.gmm =  gmm1
              print (self.gmm)
              dummy = tf.constant(-1000.0)
              if args.dataset == 'cifar':
                  self.score = tf.cast(self.gmm.log_prob(tf.cast(self.model.pen_ultimate,tf.float64)),tf.float32) / 1000
                  self.temp = tf.Variable([True,True,True,True,True,True,True,True,True,True])
                  self.t_score = tf.where(self.temp,self.score,[dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy])
              if args.dataset == 'imagenet':
                  self.score = tf.cast(self.gmm.log_prob(tf.cast(self.model.pen_ultimate,tf.float64)),tf.float32) / 10000
                  self.temp = tf.Variable([True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True])
                  self.t_score = tf.where(self.temp,self.score,[dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy])
              if args.dataset == 'gtsrb':
                  self.score = tf.cast(self.gmm.log_prob(tf.cast(self.model.pen_ultimate,tf.float64)),tf.float32) / 100
                  self.temp = tf.Variable([True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True])
                  self.t_score = tf.where(self.temp,self.score,[dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy])



              self.init_temp = tf.variables_initializer(var_list=[self.temp])
              self.assign_temp = tf.assign(self.temp[self.ind], False)
              th = tf.constant([-50.0])
              self.f1 = tf.math.less(self.t_score, th, name=None)
              self.reject = tf.cast(tf.math.reduce_all(self.f1),tf.float32)
              self.max = tf.math.reduce_max(self.t_score)
              self.loss2 = tf.math.multiply(self.reject,self.max)
              saver.restore(self.sess, model_file)
          tf.reset_default_graph()




          self.particular_score = self.score[self.ind]

      def run(self, dict_adv):
          """ Running the activation operation previously imported """
          return self.sess.run([self.model.pen_ultimate,self.model.num_correct,self.model.predictions],
                                            feed_dict=dict_adv)
      def get_features(self,dict):
          """ Running the activation operation previously imported """
          return self.sess.run([self.model.pen_ultimate,self.model.predictions],feed_dict=dict)
      def get_score(self,dict):
          return self.sess.run(self.score,feed_dict=dict)
      def get_test_score(self,dict):
          return self.sess.run([self.f1,self.t_score,self.loss2],feed_dict=dict)


class LinfPGDAttack:
  def __init__(self, Pred,GMM,P_X):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.ind = Pred.ind
    self.model = Pred.model
    self.P_CNN = P_X
    self.epsilon = 8.0
    self.num_steps = 30
    self.step_size = 1.2
    self.rand = False

    print (self.model.xent.dtype)
    print (Pred.score.dtype)
    loss = self.model.xent + Pred.loss2   # for CRopped imagenet_data
    #loss = self.model.xent + Pred.loss2       #cifar
    #loss = self.model.xent
    loss2 = -1*self.P_CNN.loss_gen_test

    self.grad = tf.gradients(loss, self.model.x_input)[0]
    self.grad2 = tf.gradients(loss2, self.P_CNN.xs)[0]
  def normalize(self,g):
      if np.max(g) > -np.min(g):
          return g/np.max(g)
      else:
          return -1 * g/np.min(g)
  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 255) # ensure valid pixel range
    else:
      x = np.copy(x_nat)

    for i in range(self.num_steps):
      grad1 = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      grad2 = P_X.sess.run(self.grad2, feed_dict={self.P_CNN.xs: (x - 127.5)/127.5,self.P_CNN.ys: y})

      grad = self.normalize(grad1) + self.normalize(grad2)
      x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 255) # ensure valid pixel range

    return x

if __name__ == "__main__":
    x,y= load_test_data()
    with open(args.GMM_dir, 'rb') as file:
        GMM = pickle.load(file)
    Predictor=CMadry_Graph(args.model_dir,GMM)
    P_X = PixelCNN(args,n_classes)

    attack = LinfPGDAttack(Predictor,GMM,P_X)

    adv_ex = []
    adv_label = []
    batch_size = 100
    num_batches = int(x.shape[0] / batch_size)
    count = 0
    if args.num_samples == -1 :
        args.num_samples = int(x.shape[0]/batch_size) * batch_size
    for i in range(args.num_samples):
        Predictor.sess.run(Predictor.init_temp,feed_dict={Predictor.ind:y[i]})
        Predictor.sess.run(Predictor.assign_temp,feed_dict={Predictor.ind:y[i]})
        x_adv_batch = attack.perturb(np.expand_dims(x[i],axis=0),np.expand_dims(y[i],axis=0),Predictor.sess)
        print ("Original Label",y[i])

        dict_predictor = {Predictor.model.x_input: np.expand_dims(x[i],axis=0)}
        x_embed,y_pred = Predictor.get_features(dict_predictor)
        score = Predictor.get_score(dict_predictor)
        print ("original  Label and feature likelihood",y_pred,score[y_pred])

        dict_predictor = {Predictor.model.x_input: np.expand_dims(x_adv_batch[0],axis=0)}
        x_embed,y_pred = Predictor.get_features(dict_predictor)
        score = Predictor.get_score(dict_predictor)
        print ("Adversarial Label and feature likelihood",y_pred,score[y_pred])
        if y[i] != y_pred:
            count = count+1
            adv_ex.append(x_adv_batch)
            adv_label.append(y_pred)

        """x[i] = (x[i] - 127.5)/ 127.5
        dict_gen = {P_X.xs:x[np.newaxis,i],P_X.ys:y[np.newaxis,i]}
        score = P_X.likelihood(dict_gen)
        print ("X likelihood",score)"""

        x_adv_batch = (x_adv_batch - 127.5)/ 127.5
        dict_gen = {P_X.xs:x_adv_batch,P_X.ys:y_pred[np.newaxis,0]}
        score = P_X.likelihood(dict_gen)
        print ("Adv X Likelihood",score)


        dict_gen = {P_X.xs:x_adv_batch,P_X.ys:y[np.newaxis,i]}
        score = P_X.likelihood(dict_gen)
        print ("Adv X Likelihood w.r.t correct label",score)

        print ("#########")
    print("Total adversarial count",count)
    adv_ex = np.array(adv_ex)
    adv_label = np.array(adv_label)




    data = {'data':np.squeeze(adv_ex), 'tlabel': y, 'alabel':np.squeeze(adv_label)}
    with open(f'{args.save_dir}/WB', 'wb') as file:
        pickle.dump(data, file)
