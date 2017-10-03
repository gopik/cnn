import tensorflow as tf


class Recognizer(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.sess = tf.Session(graph=tf.Graph()).__enter__()

        meta_graph_def = tf.saved_model.loader.load(self.sess, tags=['serve'], export_dir=self.model_dir)
        tf.summary.FileWriter('/tmp/model_eval', self.sess.graph).close()
        signature_def = meta_graph_def.signature_def[
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.x = tf.saved_model.utils.get_tensor_from_tensor_info(
            signature_def.inputs[tf.saved_model.signature_constants.PREDICT_INPUTS])
        self.y = tf.saved_model.utils.get_tensor_from_tensor_info(
            signature_def.outputs[tf.saved_model.signature_constants.PREDICT_OUTPUTS])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.__exit__(exc_type, exc_val, exec_tb=exc_tb)

    def predict(self, image):
        return self.sess.run(self.y, feed_dict={self.x: image})
