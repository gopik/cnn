from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets('MNIST-data', one_hot=True)


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


x = tf.placeholder(tf.float32, shape=[None, 784], name='train_images')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='train_labels')

# For tf.variable_scope vs tf.name_scope,
#  see https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow

with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1], name='reshaped_images')

with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope('droput'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('readout'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

summary_writer = tf.summary.FileWriter('/tmp/mnist_scratch', graph=tf.get_default_graph())

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

training_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1), name='compare_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

export_dir = '/tmp/cnn_model'
builder = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)
classification_inputs = tf.saved_model.utils.build_tensor_info(tf.placeholder(tf.string, name='tf_example'))
classification_outputs = tf.saved_model.utils.build_tensor_info(tf.constant(dtype=tf.int32, value=np.zeros(10)))
classification_output_scores = tf.saved_model.utils.build_tensor_info(tf.constant(dtype='float', value=0.0))

classification_signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs={
        tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs,
    },
    outputs={
        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs,
        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES: classification_output_scores,
    },
    method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(1):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0
            })
            print("step %d, accuracy=%f" % (i, train_accuracy))
        training_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    test_accuracy = accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
    })
    builder.add_meta_graph_and_variables(sess=session, signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature},
                                         tags=[tf.saved_model.tag_constants.SERVING])
    builder.save(as_text=True)
    print("test accuracy=%f" % test_accuracy)
