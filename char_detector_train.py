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


def get_feature(example_proto):
    parsed_feature = parse_function(example_proto)

    with tf.name_scope('decode_jpeg'):
        decoded_image = tf.image.decode_jpeg(parsed_feature['image/encoded'])
        image = tf.reshape(tf.image.rgb_to_grayscale(decoded_image, "rgb_to_grayscale"),
                           shape=[40 * 30])
        label = tf.one_hot(parsed_feature['image/class/label'], depth=37)
    return image, label


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

    parsed_feature = tf.parse_single_example(example_proto, features=features)

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


x = tf.placeholder(tf.float32, shape=[None, 1200], name='train_images')
y_ = tf.placeholder(tf.float32, shape=[None, 37], name='train_labels')

# For tf.variable_scope vs tf.name_scope,
#  see https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow

with tf.name_scope('reshape'):
    # with padding, the image size is 40x32

    x_mean = tf.reduce_mean(x, axis=0)
    x_input = tf.reshape(tf.subtract(x, x_mean, name='subract_mean'), [-1, 40, 30, 1], name='reshaped_images')

    x_image = tf.pad(x_input, [[0, 0], [0, 0], [1, 1], [0, 0]])

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
    h_pool3_flat = tf.reshape(h_pool2, [-1, 10 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder_with_default(1.0, name='keep_prob', shape=())
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('readout'):
    W_fc2 = weight_variable([1024, 37])
    b_fc2 = bias_variable([37])
    y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name='y_conv')

softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
cross_entropy = tf.reduce_mean(softmax_cross_entropy, name='cross_entropy')
global_step = tf.Variable(name='global_step', initial_value=0, dtype=tf.int32)
rate = tf.train.exponential_decay(1e-5, global_step, decay_rate=0.99, decay_steps=100)

training_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='compare_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar(name='accuracy', tensor=accuracy)
tf.summary.scalar(name='cross_entropy_loss', tensor=cross_entropy)

default_graph = tf.get_default_graph()
dataset_graph = tf.Graph()


with dataset_graph.as_default():
    train_dataset = TFRecordDataset('/tmp/fonts/out/train-00000-of-00001').map(get_feature).repeat(5).batch(
        batch_size=args.batch_size)
    val_dataset = TFRecordDataset('/tmp/fonts/out/validation-00000-of-00001').map(get_feature).batch(
        batch_size=args.batch_size)

    train_iterator = train_dataset.make_one_shot_iterator()
    train_next_batch = train_iterator.get_next()

    val_iterator = val_dataset.make_one_shot_iterator()
    val_next_batch = val_iterator.get_next()

summary_writer_train = tf.summary.FileWriter(os.path.join(args.logdir, 'train'), graph=default_graph)
summary_writer_val = tf.summary.FileWriter(os.path.join(args.logdir, 'validation'), graph=default_graph)


def export_saved_model(export_dir, session, as_text):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)
    classification_inputs = tf.saved_model.utils.build_tensor_info(x_input)
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

with tf.Session(graph=dataset_graph) as dataset_session:
    with tf.Session(graph=default_graph) as sess:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)
        else:
            sess.run(tf.global_variables_initializer())

        # val_iterator = val_dataset.make_one_shot_iterator()
        for i in range(args.num_training_steps):
            images, labels = dataset_session.run(train_next_batch)

            if i % args.checkpoint_every == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: images, y_: labels, keep_prob: 1.0
                })
                global_step_index = sess.run(global_step,
                                             feed_dict={x: images, y_: labels,
                                                        keep_prob: 1.0})
                if args.checkpoint_dir:
                    saver.save(sess, os.path.join(args.checkpoint_dir, CHECKPOINT_FILE_NAME), global_step=global_step)
                print("step %d, accuracy=%f, global_step=%d" % (i, train_accuracy, global_step_index))

            summaries, _, step_id, y_orig, y_comp, cross_entropy_val = sess.run(
                [tf.summary.merge_all(), training_step, global_step, y_, y_conv, cross_entropy],
                feed_dict={x: images, y_: labels,
                           keep_prob: 0.5})

            # print(y_orig)
            # print(y_comp)
            #
            # print(cross_entropy_val)

            summary_writer_train.add_summary(summaries, step_id)
            # validation_images, validation_labels = sess.run(val_next_batch)
            #
            # summaries, _ = sess.run([tf.summary.merge_all(), training_step],
            #                         feed_dict={x: validation_images, y_: validation_labels, keep_prob: 1.0})
            # summary_writer_val.add_summary(summaries, step_id)

        if args.save_model_dir:
            export_saved_model(args.save_model_dir, session=sess, as_text=True)
