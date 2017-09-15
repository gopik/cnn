import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser(description='Train CNN for MNIST')
parser.add_argument('--save_model_dir', help='Path to save exported model. Model will be exported only if provided')

args = parser.parse_args()

mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

with tf.Session(graph=tf.Graph()) as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, tags=['serve'], export_dir=args.save_model_dir)
    signature_def = meta_graph_def.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    x = tf.saved_model.utils.get_tensor_from_tensor_info(
        signature_def.inputs[tf.saved_model.signature_constants.PREDICT_INPUTS])
    y = tf.saved_model.utils.get_tensor_from_tensor_info(
        signature_def.outputs[tf.saved_model.signature_constants.PREDICT_OUTPUTS])
    result = sess.run(y, feed_dict={x: mnist.test.images[0].reshape(-1, 784)})
    print("Result = ", result)
    print("GT = ", mnist.test.labels[0])
