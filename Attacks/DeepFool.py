import copy
import logging
import warnings
import math

import numpy as np
import tensorflow as tf

from cleverhans.attacks import Attack
from cleverhans.model import Model, wrapper_warning_logits, CallableModelWrapper
from cleverhans import utils
from cleverhans import utils_tf

np_dtype = np.dtype("float32")

_logger = utils.create_logger("cleverhans.attacks.deep_fool")
_logger.setLevel(logging.INFO)


class DF(object):
    def __init__(self, sess, model,nb_class,batch_size,x):
        """:x: input placeholder
        """
        self.sess = sess
        self.x = x
        self.nb_candidate = 10
        self.overshoot = 0.02
        self.max_iter = 10
        self.clip_min = 0
        self.clip_max = 255
        self.nb_classes = nb_class
        self.batch_size = batch_size
        #creating graph
        from cleverhans.attacks_tf import jacobian_graph
        # Define graph wrt to this input placeholder
        self.logits = model.get_logits(x)
        self.nb_classes = self.logits.get_shape().as_list()[-1]
        assert (
            self.nb_candidate <= self.nb_classes
        ), "nb_candidate should not be greater than nb_classes"
        self.preds = tf.reshape(
            tf.nn.top_k(self.logits, k=self.nb_candidate)[0], [-1, self.nb_candidate]
        )
        # grads will be the shape [batch_size, nb_candidate, image_size]
        self.grads = tf.stack(jacobian_graph(self.preds, self.x, self.nb_candidate), axis=1)
    def attack(self,X):
        X_adv = []
        num_eval_examples = X.shape[0]
        eval_batch_size = self.batch_size
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}'.format(bend - bstart))
            if ibatch == 0:
                X_adv = self.attack_batch(X[bstart:bend])
            else:
                X_adv = np.concatenate((X_adv,self.attack_batch(X[bstart:bend])),axis=0)
        return X_adv

    def attack_batch(self,sample,feed=None):
        """
        TensorFlow implementation of DeepFool.
        Paper link: see https://arxiv.org/pdf/1511.04599.pdf
        :param sess: TF session
        :param x: The input placeholder
        :param predictions: The model's sorted symbolic output of logits, only the
                           top nb_candidate classes are contained
        :param logits: The model's unnormalized output tensor (the input to
                       the softmax layer)
        :param grads: Symbolic gradients of the top nb_candidate classes, procuded
                     from gradient_graph
        :param sample: Numpy array with batch of samples input
        :param nb_candidate: The number of classes to test against, i.e.,
                             deepfool only consider nb_candidate classes when
                             attacking(thus accelerate speed). The nb_candidate
                             classes are chosen according to the prediction
                             confidence during implementation.
        :param overshoot: A termination criterion to prevent vanishing updates
        :param max_iter: Maximum number of iteration for DeepFool
        :param clip_min: Minimum value for components of the example returned
        :param clip_max: Maximum value for components of the example returned
        :return: Adversarial examples"""


        adv_x = copy.copy(sample)
        # Initialize the loop variables
        iteration = 0
        current = utils_tf.model_argmax(self.sess, self.x, self.logits, adv_x, feed=feed)
        if current.shape == ():
            current = np.array([current])
        w = np.squeeze(np.zeros(sample.shape[1:]))  # same shape as original image
        r_tot = np.zeros(sample.shape)
        original = current  # use original label as the reference

        _logger.debug("Starting DeepFool attack up to %s iterations", self.max_iter)
        # Repeat this main loop until we have achieved misclassification
        while np.any(current == original) and iteration < self.max_iter:

            if iteration % 5 == 0 and iteration > 0:
                print("Attack result at iteration %s is %s", iteration, current)
            gradients = self.sess.run(self.grads, feed_dict={self.x: adv_x})
            predictions_val = self.sess.run(self.preds, feed_dict={self.x: adv_x})
            for idx in range(sample.shape[0]):
                pert = np.inf
                if current[idx] != original[idx]:
                    continue
                for k in range(1, self.nb_candidate):
                    w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                    f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                    # adding value 0.00001 to prevent f_k = 0
                    pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten())
                    if pert_k < pert:
                        pert = pert_k
                        w = w_k
                r_i = pert * w / np.linalg.norm(w)
                r_tot[idx, ...] = r_tot[idx, ...] + r_i

            adv_x = np.clip(r_tot + sample, self.clip_min, self.clip_max)
            current = utils_tf.model_argmax(self.sess, self.x, self.logits, adv_x, feed=feed)
            if current.shape == ():
                current = np.array([current])
            # Update loop variables
            iteration = iteration + 1

        # need more revision, including info like how many succeed
        _logger.info("Attack result at iteration %s is %s", iteration, current)
        _logger.info(
            "%s out of %s become adversarial examples at iteration %s",
            sum(current != original),
            sample.shape[0],
            iteration,
        )
        # need to clip this image into the given range
        adv_x = np.clip((1 + self.overshoot) * r_tot + sample, self.clip_min, self.clip_max)
        return np.asarray(adv_x, dtype=np_dtype)
