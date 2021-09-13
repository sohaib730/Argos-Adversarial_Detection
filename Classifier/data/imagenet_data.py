"""
Utilities for loading the small ImageNet dataset used in Oord et al.
use scripts/png_to_npz.py to create the npz files

The code here currently assumes that the preprocessing was done manually.
TODO: make automatic and painless
"""

"""
ImageNet 32x32 DataLoader for training with augmentation
"""

import numpy as np
import os
import pickle
import sys
import tensorflow as tf

version = sys.version_info



class DataLoader(object):
    """
    Unpickles the CIFAR10 dataset from a specified folder containing a pickled
    version following the format of Krizhevsky which can be found
    [here](https://www.cs.toronto.edu/~kriz/cifar.html).

    Inputs to constructor
    =====================

        - path: path to the pickled dataset. The training data must be pickled
        into five files named data_batch_i for i = 1, ..., 5, containing 10,000
        examples each, the test data
        must be pickled into a single file called test_batch containing 10,000
        examples, and the 10 class names must be
        pickled into a file called batches.meta. The pickled examples should
        be stored as a tuple of two objects: an array of 10,000 32x32x3-shaped
        arrays, and an array of their 10,000 true labels.

    """
    def __init__(self, path):
        train_filenames = ['batch_{}'.format(ii+1) for ii in range(1)]
        eval_filename = 'val'

        samples = 40517
        train_images = np.zeros((samples, 32, 32, 3), dtype='uint8')
        train_labels = np.zeros(samples, dtype='int32')
        for ii, fname in enumerate(train_filenames):
            cur_images,cur_labels = self._load_datafile(os.path.join(path, fname))

            train_images[ii * samples : (ii+1) * samples, ...] = cur_images
            train_labels[ii * samples : (ii+1) * samples, ...] = cur_labels
        eval_images, eval_labels = self._load_datafile(
            os.path.join(path, eval_filename))
        """self.label_names = data_dict[b'label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')"""
        perm = np.random.permutation(train_images.shape[0])
        self.train_data = DataSubset(train_images, train_labels,perm)
        perm = range(eval_images.shape[0])
        self.eval_data = DataSubset(eval_images, eval_labels,perm)

    @staticmethod
    def _load_datafile(filename):
      with open(filename, 'rb') as fo:
          if version.major == 3:
              data_dict = pickle.load(fo, encoding='bytes')
          else:
              data_dict = pickle.load(fo)

          assert data_dict['x'].dtype == np.uint8
          #image_data = data_dict['data']

          return data_dict['x'], data_dict['y']

class AugmentedImageNetData(object):
    """
    Data augmentation wrapper over a loaded dataset.

    Inputs to constructor
    =====================
        - raw_cifar10data: the loaded CIFAR10 dataset, via the CIFAR10Data class
        - sess: current tensorflow session
        - model: current model (needed for input tensor)
    """
    def __init__(self, raw_cifar10data, sess, model):
        #assert isinstance(raw_cifar10data, CIFAR10Data_Edit1)
        self.image_size = 32

        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
            img, self.image_size + 4, self.image_size + 4),
            self.x_input_placeholder)
        cropped = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size,
                                                             self.image_size,
                                                             3]), padded)
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
        #flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img))
        self.augmented = flipped

        self.train_data = AugmentedDataSubset(raw_cifar10data.train_data, sess,
                                             self.x_input_placeholder,
                                              self.augmented)

        self.eval_data = AugmentedDataSubset(raw_cifar10data.eval_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)



class DataSubset(object):
    def __init__(self, xs, ys,r_perm):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = r_perm

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=False):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += batch_size
        return batch_xs, batch_ys


class AugmentedDataSubset(object):
    def __init__(self, raw_datasubset, sess, x_input_placeholder,
                 augmented):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=False):
        raw_batch = self.raw_datasubset.get_next_batch(batch_size, multiple_passes,
                                                       reshuffle_after_pass)
        images = raw_batch[0].astype(np.float32)
        return self.sess.run(self.augmented, feed_dict={self.x_input_placeholder:
                                                    raw_batch[0]}), raw_batch[1]
