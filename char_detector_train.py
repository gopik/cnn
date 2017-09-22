import argparse
import os.path

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data.python.ops.dataset_ops import TFRecordDataset

parser = argparse.ArgumentParser(description='Train CNN for MNIST')
parser.add_argument('--save_model_dir', help='Path to save exported model. Model will be exported only if provided')
parser.add_argument('--num_training_steps', type=int, default=1000, help='Number of steps to run the training for')
parser.add_argument('--checkpoint_dir',
                    help='Path to load/save checkpoint. If provided, latest checkpoint will be loaded from here')
parser.add_argument('--checkpoint_every', type=int, help='Num iterations to checkpoint after', default=100)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--logdir', default='/tmp/cnn/logs')

args = parser.parse_args()


def parse_function(example_proto):
    features = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/channels': tf.FixedLenFeature([], tf.int64),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
    }

    parsed_feature = tf.parse_example(example_proto, features=features)
    return parsed_feature


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
    # with padding, the image size is 40x32
    x_image = tf.pad(tf.reshape(x, [-1, 40, 30, 1], name='reshaped_images'), [[0, 0], [0, 0], [1, 1], [0, 0]])

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
    W_fc1 = weight_variable([10 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder_with_default(1.0, name='keep_prob', shape=())
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('readout'):
    W_fc2 = weight_variable([1024, 36])
    b_fc2 = bias_variable([36])
    y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name='y_conv')

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
    classification_inputs = tf.saved_model.utils.build_tensor_info(x)
    classification_output_scores = tf.saved_model.utils.build_tensor_info(y_conv)

    classification_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            tf.saved_model.signature_constants.PREDICT_INPUTS: classification_inputs,
        },
        outputs={
            tf.saved_model.signature_constants.PREDICT_OUTPUTS: classification_output_scores,
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    builder.add_meta_graph_and_variables(sess=session, signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature},
                                         tags=[tf.saved_model.tag_constants.SERVING])
    builder.save(as_text=as_text)


saver = tf.train.Saver()

CHECKPOINT_FILE_NAME = 'checkpoint'

train_dataset = TFRecordDataset('/tmp/fonts/out/train-00000-of-000001').map(parse_function)
val_dataset = TFRecordDataset('/tmp/fonts/out/val-00000-of-000001').map(parse_function)

val_images_count = len(val_dataset['image/encoded'])

with tf.Session() as sess:
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    if latest_checkpoint:
        saver.restore(sess, latest_checkpoint)
    else:
        sess.run(tf.global_variables_initializer())

    for i in range(args.num_training_steps):
        batch = train_dataset.batch(args.batch_size)
        if i % args.checkpoint_every == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch['image/encoded'], y_: batch['image/class/label'], keep_prob: 1.0
            })
            global_step_index = sess.run(global_step,
                                         feed_dict={x: batch['image/encoded'], y_: batch['image/class/label'],
                                                    keep_prob: 0.5})
            if args.checkpoint_dir:
                saver.save(sess, os.path.join(args.checkpoint_dir, CHECKPOINT_FILE_NAME), global_step=global_step)
            print("step %d, accuracy=%f, global_step=%d" % (i, train_accuracy, global_step_index))

        summaries, _, step_id = sess.run([tf.summary.merge_all(), training_step, global_step],
                                         feed_dict={ x: batch['image/encoded'], y_: batch['image/class/label'], keep_prob: 0.5})
        summary_writer_train.add_summary(summaries, step_id)
        # Sample 100 images from validation
        val_indices = np.random.choice(np.arange(val_images_count), 100, replace=False)
        validation_images = val_dataset[val_indices]['image/encoded']
        validation_labels = val_dataset[val_indices]['image/class/label']

        summaries, _ = sess.run([tf.summary.merge_all(), training_step],
                                feed_dict={x: validation_images, y_: validation_labels, keep_prob: 1.0})
        summary_writer_val.add_summary(summaries, step_id)

    if args.save_model_dir:
        export_saved_model(args.save_model_dir, session=sess, as_text=True)
