from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import numpy as np

from tensorflow.python.layers.core import Dense

tf.logging.set_verbosity(tf.logging.INFO)


# Our application logic will be added here

def cnn_model_fn(features, labels, mode):
    """Model function for the CNN"""

    # Input Layer
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    # Convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, shape=(-1, 7 * 7 * 64))
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    dense_layer = Dense(units=10, activation=tf.nn.relu)

    logits = dense_layer.apply(dropout)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                          export_outputs=dict(
                                              eval=tf.estimator.export.ClassificationOutput(
                                                  classes=tf.as_string(predictions['classes']))))

    # Computing loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits
    )

    accuracy, update_op = tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])

    if mode == tf.estimator.ModeKeys.TRAIN:
        v = dense_layer.variables
        for i in range(len(v)):
            tf.summary.histogram('dense_weights%d' % i, dense_layer.variables[i])
        tf.summary.scalar("train_loss", loss)
        tf.summary.scalar("training_accuracy1", accuracy)
        tf.summary.scalar("training_accuracy2", update_op)

        summary_hook = tf.train.SummarySaverHook(save_steps=10,
                                                 output_dir='/tmp/1-test-mnist_cnn_training',
                                                 summary_op=tf.summary.merge_all())

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op,
                                          training_hooks=[summary_hook])

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels,
                                        predictions=predictions['classes'])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                              model_dir='/tmp/test-mnist_cnn_model')

    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(tensors_to_log,
    #                                           every_n_iter=5)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data}, y=train_labels, batch_size=100, num_epochs=None,
        shuffle=True
    )
    feature_spec = {
        "x": tf.placeholder(shape=(None, 784*3), dtype='float', name='input_placeholder')
    }

    def input_receiver_fn():
        return tf.estimator.export.ServingInputReceiver(features={'x': tf.placeholder(dtype='float', shape=(None, 784*3))},
                                                        receiver_tensors=tf.as_string(train_data))


    # mnist_classifier.train(input_fn=train_input_fn, steps=1)
    mnist_classifier.export_savedmodel(export_dir_base="/tmp/model", serving_input_receiver_fn=input_receiver_fn,
                                       as_text=True)
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={'x': eval_data},
    #     y=eval_labels,
    #     num_epochs=1,
    #     shuffle=False
    # )
    #
    # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    #
    # print(eval_results)


if __name__ == "__main__":
    tf.app.run(main=main)
