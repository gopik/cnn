import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():
    parser = argparse.ArgumentParser(description='Train CNN for MNIST')
    parser.add_argument('--save_model_dir', help='Path to save exported model. Model will be exported only if provided')
    args = parser.parse_args()
    mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

    with Recognizer(args.save_model_dir) as recognizer:
        print(recognizer.predict(mnist.test.images[0]))


class Recognizer(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def __enter__(self):
        self.sess = tf.Session(graph=tf.Graph()).__enter__()
        meta_graph_def = tf.saved_model.loader.load(self.sess, tags=['serve'], export_dir=self.model_dir)
        signature_def = meta_graph_def.signature_def[
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.x = tf.saved_model.utils.get_tensor_from_tensor_info(
            signature_def.inputs[tf.saved_model.signature_constants.PREDICT_INPUTS])
        self.y = tf.saved_model.utils.get_tensor_from_tensor_info(
            signature_def.outputs[tf.saved_model.signature_constants.PREDICT_OUTPUTS])

        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.__exit__(exc_type, exc_val, exec_tb=exc_tb)

    def predict(self, image):
        return self.sess.run(self.y, feed_dict={self.x : image.reshape(-1, 784)})


if __name__ == '__main__':
    main()
