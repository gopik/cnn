import functools
import os.path

import numpy as np
import tensorflow as tf

from char_detector_train import mean_dataset


def compute_mean_image():
    def reduce_mean(acc, data):
        cur_mean, count = acc
        val, _ = data
        return cur_mean * count/(count + 1) + val/(count + 1), count + 1

    if os.path.exists('/tmp/mean_image.npy'):
        return np.load('/tmp/mean_image.npy')

    print('Save mean not found, computing ...')
    mean, _ = functools.reduce(reduce_mean, mean_dataset, (0.0, 0))
    np.save('/tmp/mean_image.npy', mean)
    print('Saving computed mean image')
    return mean


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial, name='weight_var')


def bias_variable(shape):
    initial = tf.constant(name='biases', value=0.1, shape=shape)
    return tf.Variable(initial, name='bias_var')


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')