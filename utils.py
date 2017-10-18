import functools
import os.path

import numpy as np
import tensorflow as tf


def compute_mean_image(mean_dataset):
    def reduce_mean(acc, data):
        cur_mean, count = acc
        val, _ = data
        return cur_mean * count / (count + 1) + val / (count + 1), count + 1

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


def get_padding(h, w, target_h, target_w):
    times_h = np.ceil(h / target_h)
    times_w = np.ceil(w / target_w)

    times = max(times_h, times_w)
    h_pad = int(target_h * times - h)
    w_pad = int(target_w * times - w)

    if w_pad < 0:
        print(h, w, target_h, target_w)
    h_pad_top, _ = divmod(h_pad, 2)
    w_pad_left, _ = divmod(w_pad, 2)

    return (h_pad_top, h_pad - h_pad_top), (w_pad_left, w_pad - w_pad_left)