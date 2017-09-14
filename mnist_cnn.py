from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import argparse
import os.path

parser = argparse.ArgumentParser(description='Train CNN for MNIST')
parser.add_argument('--save_model_dir', help='Path to save exported model. Model will be exported only if provided')
parser.add_argument('--num_training_steps', type=int, default=1000, help='Number of steps to run the training for')
parser.add_argument('--checkpoint_dir',
                    help='Path to load/save checkpoint. If provided, latest checkpoint will be loaded from here')
parser.add_argument('--checkpoint_every', type=int, help='Num iterations to checkpoint after', default=100)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--logdir', default='/tmp/cnn/logs')

args = parser.parse_args()

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
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('readout'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv), name='cross_entropy')
global_step = tf.Variable(name='global_step', initial_value=0, dtype=tf.int32)
rate = tf.train.exponential_decay(1e-4, global_step, decay_rate=0.99, decay_steps=100)

training_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='compare_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar(name='accuracy', tensor=accuracy)
tf.summary.scalar(name='cross_entropy_loss', tensor=cross_entropy)

summary_writer_train = tf.summary.FileWriter(os.path.join(args.logdir, 'train'), graph=tf.get_default_graph())
summary_writer_val = tf.summary.FileWriter(os.path.join(args.logdir, 'validation'), graph=tf.get_default_graph())


def export_saved_model(export_dir, session, as_text):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)
    classification_inputs = tf.saved_model.utils.build_tensor_info(tf.placeholder(tf.string))
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

    builder.add_meta_graph_and_variables(sess=session, signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature},
                                         tags=[tf.saved_model.tag_constants.SERVING])
    builder.save(as_text=as_text)


saver = tf.train.Saver()

CHECKPOINT_FILE_NAME = 'checkpoint'

val_images_count = len(mnist.validation.images)

with tf.Session() as sess:
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    if latest_checkpoint:
        saver.restore(sess, latest_checkpoint)
    else:
        sess.run(tf.global_variables_initializer())

    for i in range(args.num_training_steps):
        batch = mnist.train.next_batch(args.batch_size)
        if i % args.checkpoint_every == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0
            })
            global_step_index = sess.run(global_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if args.checkpoint_dir:
                saver.save(sess, os.path.join(args.checkpoint_dir, CHECKPOINT_FILE_NAME), global_step=global_step)
            print("step %d, accuracy=%f, global_step=%d" % (i, train_accuracy, global_step_index))

        summaries, _, step_id = sess.run([tf.summary.merge_all(), training_step, global_step],
                                         feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})
        summary_writer_train.add_summary(summaries, step_id)
        # Sample 100 images from validation
        val_indices = np.random.choice(np.arange(val_images_count), 100, replace=False)
        validation_images = mnist.validation.images[val_indices]
        validation_labels = mnist.validation.labels[val_indices]

        summaries, _ = sess.run([tf.summary.merge_all(), training_step],
                                feed_dict={x: validation_images, y_: validation_labels, keep_prob: 1.0})
        summary_writer_val.add_summary(summaries, step_id)

    test_accuracy = accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
    })

    print("test accuracy=%f" % test_accuracy)
    if args.save_model_dir:
        export_saved_model(args.save_model_dir, session=sess, as_text=True)
