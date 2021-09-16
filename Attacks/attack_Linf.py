
"""L-inf attacks using Cleverhans Library
 __author__ = Sohaib Kiani
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sys
import argparse
import time
import numpy as np
import logging

import tensorflow as tf
from tensorflow.python.platform import app, flags
from cleverhans.utils import set_log_level, to_categorical
from cleverhans.utils_tf import model_eval

from PIL import Image
import math
import pickle
#import eagerpy as ep
args = ''
nb_classes = 0


def load_data():

    sys.path.append(os.path.join( os.getcwd(), '..', 'Classifier/data' ))
    if args.dataset == 'cifar':
        import cifar10_data
        cifar = cifar10_data.CIFAR10Data(args.data_dir)
        assert nb_classes == 10
        assert cifar.eval_data.n == 10000
    elif args.dataset == 'gtsrb':
        import GTSRB_data
        cifar = GTSRB_data.DataLoader(args.data_dir)
        assert nb_classes == 43
    elif args.dataset == 'imagenet':
        import imagenet_data
        cifar = imagenet_data.DataLoader(args.data_dir)
        assert nb_classes == 16
        print (cifar.eval_data.n)
        #assert cifar.eval_data.n == 1181
    return cifar

def load_model(model_file,sess):
    from model_wrapper_Cleverhans import make_wresnet
    model = make_wresnet(nb_classes=nb_classes)
    saver = tf.train.Saver()
    # Restore the checkpoint
    saver.restore(sess, model_file)
    print ("Loaded Model")
    SCOPE = "cifar10_challenge"
    model2 = make_wresnet(scope=SCOPE,nb_classes=nb_classes)
    assert len(model.get_vars()) == len(model2.get_vars())
    found = [False] * len(model2.get_vars())
    for var1 in model.get_vars():
      var1_found = False
      var2_name = SCOPE + "/" + var1.name
      for idx, var2 in enumerate(model2.get_vars()):
        if var2.name == var2_name:
          var1_found = True
          found[idx] = True
          sess.run(tf.assign(var2, var1))
          break
      assert var1_found, var1.name
    assert all(found)

    model = model2
    #saver = tf.train.Saver()


    """modifier = tf.Variable(np.zeros((100,32,32,3), dtype=np.dtype('float32')))
    init = tf.variables_initializer(var_list=modifier)
    sess.run(init)
    print ("DONE")"""
    return model

def run_attack(sess, model, cifar,x,y,attack_type,eps):
    X_test = cifar.eval_data.xs
    print ("Range of data should be 0-255 and actual is: ",str(np.min(X_test))+" "+str(np.max(X_test)))
    Y_test = to_categorical(cifar.eval_data.ys, nb_classes)
    assert Y_test.shape[1] == nb_classes
    print ("test data shape",X_test.shape)

    set_log_level(logging.DEBUG)



    attack_params = {'batch_size': args.batch_size,
                     'clip_min': 0., 'clip_max': 255.}



    if attack_type == 'fgsm':
        print ("Running FGM attack")
        attack_params.update({'eps':eps,'ord':np.inf})
        from cleverhans.attacks import FastGradientMethod
        attacker = FastGradientMethod(model, sess=sess)

    elif  attack_type == 'pgd':
        attack_params.update({'eps':eps,'ord':np.inf, 'nb_iter': 20,'eps_iter':0.8 })
        from cleverhans.attacks import ProjectedGradientDescent
        attacker = ProjectedGradientDescent(model, sess=sess)
    elif attack_type == 'deepFool':
        attack_params.update({'max_iter': 20})
        from cleverhans.attacks import DeepFool
        attacker = DeepFool(model, sess=sess)
    elif attack_type == 'mim':
        attack_params.update({'eps':eps,'ord':np.inf, 'nb_iter': 20,'eps_iter':0.8 })
        from cleverhans.attacks import MomentumIterativeMethod
        attacker = MomentumIterativeMethod(model, sess=sess)
    elif attack_type == 'stm':
        #attack_params.update({'eps':2000.0,'nb_iter':20})
        from cleverhans.attacks import SpatialTransformationMethod
        attacker = SpatialTransformationMethod(model, sess=sess)
    elif attack_type == 'enm':
        from cleverhans.attacks import ElasticNetMethod
        attacker = ElasticNetMethod(model, sess=sess)



    eval_par = {'batch_size': args.batch_size}


    t1 = time.time()
    x_adv = attacker.generate(x, **attack_params)
    preds_adv = model.get_probs(x_adv)
    logits  = model.get_logits(x)
    #print (len(x_adv))

    num_eval_examples =  args.num_samples #cifar.eval_data.n
    eval_batch_size = args.batch_size
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv_all = [] # adv accumulator
    y_adv_all = []
    y_true = []
    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)
        print('batch size: {}'.format(bend - bstart))

        x_batch = X_test[bstart:bend, :]
        y_batch = Y_test[bstart:bend]
        y_clean = np.argmax(sess.run (logits,feed_dict={x:x_batch}),axis=1)
        x_b_adv,pred = sess.run([x_adv,preds_adv],feed_dict = {x:x_batch,y:y_batch})
        y_b_adv = np.argmax(sess.run (logits,feed_dict={x:x_b_adv}),axis=1)

        count = 0
        y_batch = np.argmax(y_batch,axis=1)
        for i in range(eval_batch_size):
            if (y_b_adv[i] != y_batch[i] and y_clean[i] == y_batch[i]):
                l_inf = np.amax(np.abs(x_batch[i] - x_b_adv[i]))
                x_adv_all.append(x_b_adv[i])
                y_adv_all.append(y_b_adv[i])
                y_true.append(y_batch[i])
                count +=1
        #print (y_adv_all)
        #print (y_true)
        print (f"Totat adversarial {count} in this batch")



    x_adv_all = np.array(x_adv_all)
    y_true = np.array(y_true)
    y_adv_all = np.array(y_adv_all)

    print ('Adv Label',y_adv_all[0:20])
    print ('Ori Label',y_true[0:20])

    #y_adv = np.squeeze(y_adv)
    print (x_adv_all.shape)
    print (y_adv_all.shape)
    print (y_true.shape)


    count = 0

    print (f"Total adversarial examples found {x_adv_all.shape[0]} out of {num_eval_examples}")
    data = {'data':x_adv_all,'tlabel':y_true,'alabel':y_adv_all}

    save_path = os.path.join(args.save_dir,attack_type)
    print ("Saving Attack",f'{save_path}_E{eps}')
    pickle.dump(data, open(f'{save_path}_E{eps}', 'wb'))

    t2 = time.time()
    print("Took", t2 - t1, "seconds")
    print ("Range of data should be 0-255 and actual is: ",str(np.min(x_adv_all))+" "+str(np.max(x_adv_all)))
    """image=((x_adv_all[2])).astype(np.uint8)
    img=Image.fromarray(image)
    img.save(f"{args.dataset}_{attack_type}.jpeg")"""



if __name__ == '__main__':



  parser = argparse.ArgumentParser()
  # data I/O
  parser.add_argument('-d', '--dataset', type=str, default='cifar', help='Can be either cifar|imagenet|gtsrb')
  parser.add_argument('-n', '--num_samples', type=int, default=1600, help='How many test samples to be attacked')
  parser.add_argument('-b', '--batch_size', type=int, default=100, help='Batch Size')

  parser.add_argument('-i', '--data_dir', type=str, default='', help='Location for the dataset')
  parser.add_argument('-o', '--save_dir', type=str, default='', help='Location for the dataset')
  parser.add_argument('-m', '--model_dir', type=str, default='/Classifier/Model/ResNet50_ckpt', help='Classifier Model')
  args = parser.parse_args()

  args.data_dir = os.path.join( os.getcwd(), '..', f'data/{args.dataset}' )
  args.save_dir = os.path.join( os.getcwd(), '..', f'data/{args.dataset}/Adversarial' )
  if not os.path.exists(f"{args.save_dir}"):
      os.makedirs(f"{args.save_dir}")
  args.model_dir = os.path.join( os.getcwd(), '..', f'Classifier/{args.dataset}_Model/ResNet50_ckpt' )

  ### Model, Source and Save Dataset dir


  if args.dataset == 'cifar':
        nb_classes = 10
  if args.dataset == 'gtsrb':
        nb_classes = 43
  if args.dataset == 'imagenet':
        nb_classes = 16



  x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  #flags.DEFINE_string('checkpoint_dir', default_ckpt_dir, 'Checkpoint directory to load')

  #flags.DEFINE_string('dataset_dir', default_data_dir, 'Dataset directory')
  #flags.DEFINE_string('save_dir', save_dir, 'Dataset directory')


  model_file = tf.train.latest_checkpoint(args.model_dir)
  if model_file is None:
    print('No model found')
    sys.exit()
  data = load_data()

  sess = tf.Session()
  model = load_model(model_file,sess)





  attack_type = ['fgsm','pgd','mim']
  eps = [4,8,16]

  for attack in attack_type:
      for e in eps:
          run_attack(sess,model,data,x,y,attack,e)


  sess.close()
