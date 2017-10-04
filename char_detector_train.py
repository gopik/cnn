import argparse
import os.path

import tensorflow as tf

from tf_image_reader import TFImageReader
from utils import compute_mean_image, weight_variable, bias_variable, conv2d, max_pool_2x2

parser = argparse.ArgumentParser(description='Train CNN for MNIST')
parser.add_argument('--save_model_dir', help='Path to save exported model. Model will be exported only if provided')
parser.add_argument('--num_training_steps', type=int, default=1000, help='Number of steps to run the training for')
parser.add_argument('--checkpoint_dir',
                    help='Path to load/save checkpoint. If provided, latest checkpoint will be loaded from here')
parser.add_argument('--checkpoint_every', type=int, help='Num iterations to checkpoint after', default=100)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--logdir', default='/tmp/cnn/logs')
parser.add_argument('--dropout_keep_ratio', type=float, default=0.5)
parser.add_argument('--train_dataset', default='/tmp/fonts/small_tx/out/train-00000-of-00001')
parser.add_argument('--validation_dataset', default='/tmp/fonts/small_tx/out/validation-00000-of-00001')


args = parser.parse_args()

default_graph = tf.Graph()

train_dataset = TFImageReader(args.train_dataset, args.batch_size, unlimited=True)
val_dataset = TFImageReader(args.validation_dataset, args.batch_size, unlimited=True)
mean_dataset = TFImageReader(args.train_dataset, 1)

# invert mean image
image_mean = 255 - compute_mean_image(mean_dataset)

# For tf.variable_scope vs tf.name_scope,
#  see https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow

with default_graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 1200], name='train_images')
    y_ = tf.placeholder(tf.float32, shape=[None, 37], name='train_labels')

    mean_image = tf.reshape(tf.constant(image_mean, dtype=tf.float32), [-1, 40, 30, 1], name='reshaped_mean_image')

    with tf.name_scope('reshape'):
        # invert input image
        x_input = 255 - tf.reshape(x, [-1, 40, 30, 1], name='reshaped_images')

        x_image_sub_mean = x_input - mean_image
        x_image = tf.pad(x_image_sub_mean, [[0, 0], [0, 0], [1, 1], [0, 0]])

    conv_keep_prob = tf.placeholder_with_default(1.0, name='conv_keep_prob', shape=())
    max_norm = tf.constant(4.0)

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 36])
        b_conv1 = bias_variable([36])

        W_conv1_norm = tf.clip_by_norm(W_conv1, max_norm, axes=[1, 2])
        h_conv1 = tf.nn.dropout(tf.nn.relu(conv2d(x_image, W_conv1_norm) + b_conv1), conv_keep_prob)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 36, 32])
        b_conv2 = bias_variable([32])

        W_conv2_norm = tf.clip_by_norm(W_conv2, max_norm, axes=[1, 2])
        h_conv2 = tf.nn.dropout(tf.nn.relu(conv2d(h_pool1, W_conv2_norm) + b_conv2), conv_keep_prob)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # with tf.name_scope('conv3'):
    #     W_conv3 = weight_variable([3, 3, 32, 64])
    #     b_conv3 = bias_variable([64])
    #     h_conv3 = tf.nn.dropout(tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3), conv_keep_prob)
    #
    # with tf.name_scope('pool3'):
    #     h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([10 * 8 * 32, 1024])
        b_fc1 = bias_variable([1024])
        h_pool3_flat = tf.reshape(h_pool2, [-1, 10 * 8 * 32])

        W_fc1_norm = tf.clip_by_norm(W_fc1, max_norm, axes=[1])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1_norm) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder_with_default(1.0, name='keep_prob', shape=())
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('readout'):
        W_fc2 = weight_variable([1024, 37])
        b_fc2 = bias_variable([37])

        W_fc2_norm = tf.clip_by_norm(W_fc2, max_norm, axes=[1])
        y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2_norm), b_fc2, name='y_conv')

    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(softmax_cross_entropy, name='cross_entropy')
    global_step = tf.Variable(name='global_step', initial_value=0, dtype=tf.int32, trainable=False)
    rate = tf.train.exponential_decay(1e-4, global_step, decay_rate=0.99, decay_steps=100)

    optimizer = tf.train.AdamOptimizer(1e-4)
    gradients = optimizer.compute_gradients(cross_entropy)
    training_step = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='compare_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar(name='accuracy', tensor=accuracy)
    tf.summary.scalar(name='cross_entropy_loss', tensor=cross_entropy)

    w_conv1_r1 = tf.concat(tf.unstack(tf.reshape(W_conv1, [6, 6, 3, 3, 1]), axis=1), axis=2)
    w_conv1_r1 = tf.stack([tf.concat(tf.unstack(w_conv1_r1), axis=0)])

    tf.summary.image(name='W_conv1_weights', tensor=w_conv1_r1, max_outputs=6)

    conv_grad = None
    for (g, v) in gradients:
        if v == W_conv1:
            conv_grad = g
            break

    tf.summary.histogram(name='conv_image_grad', values=conv_grad)

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


with default_graph.as_default():
    saver = tf.train.Saver()

CHECKPOINT_FILE_NAME = 'checkpoint'

with tf.Session(graph=default_graph) as sess:
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    if latest_checkpoint:
        saver.restore(sess, latest_checkpoint)
    else:
        sess.run(tf.global_variables_initializer())

    validation_dataset_iterator = iter(val_dataset)
    for i, (images, labels) in zip(range(args.num_training_steps), train_dataset):
        if i % args.checkpoint_every == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: images, y_: labels, keep_prob: 1.0
            })
            global_step_index = sess.run(global_step,
                                         feed_dict={x: images, y_: labels})
            if args.checkpoint_dir:
                saver.save(sess, os.path.join(args.checkpoint_dir, CHECKPOINT_FILE_NAME), global_step=global_step)
            print("step %d, accuracy=%f, global_step=%d" % (i, train_accuracy, global_step_index))

        summaries, _, step_id, y_orig, y_comp, cross_entropy_val = sess.run(
            [tf.summary.merge_all(), training_step, global_step, y_, y_conv, cross_entropy],
            feed_dict={x: images, y_: labels,
                       keep_prob: args.dropout_keep_ratio, conv_keep_prob: 0.5})

        summary_writer_train.add_summary(summaries, step_id)

        validation_images, validation_labels = next(validation_dataset_iterator)
        summaries = sess.run(tf.summary.merge_all(),
                             feed_dict={x: validation_images, y_: validation_labels})
        summary_writer_val.add_summary(summaries, step_id)

    if args.save_model_dir:
        export_saved_model(args.save_model_dir, session=sess, as_text=True)
