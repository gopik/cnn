import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser(description='Train CNN for MNIST')
parser.add_argument('--save_model_dir', help='Path to save exported model. Model will be exported only if provided')

args = parser.parse_args()

mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

with tf.Session(graph=tf.Graph()) as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, tags=['serve'], export_dir=args.save_model_dir)
    x = tf.get_default_graph().get_tensor_by_name('train_images:0')
    y_conv = tf.get_default_graph().get_tensor_by_name('readout/y_conv:0')
    keep_probability = tf.get_default_graph().get_tensor_by_name('dropout/keep_prob:0')

    result = sess.run(y_conv, feed_dict={x: mnist.test.images[0].reshape(-1, 784), keep_probability:1.0})
    print("Result = ", result)
    print("GT = ", mnist.test.labels[0])
