import argparse
import os.path
from tf_image_reader import TFImageReader

import tensorflow as tf

from utils import weight_variable, bias_variable, conv2d, max_pool_2x2, salt_and_pepper

parser = argparse.ArgumentParser(description='Train CNN for MNIST')
parser.add_argument('--save_model_dir', help='Path to save exported model. Model will be exported only if provided')
parser.add_argument('--num_training_steps', type=int, default=1000, help='Number of steps to run the training for')
parser.add_argument('--checkpoint_dir',
                    help='Path to load/save checkpoint. If provided, latest checkpoint will be loaded from here')
parser.add_argument('--checkpoint_every', type=int, help='Num iterations tocheckpoint after', default=1000)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--logdir', default='/tmp/cnn/logs')
parser.add_argument('--dropout_keep_ratio', type=float, default=0.6)
parser.add_argument('--conv_dropout_keep_ratio', type=float, default=0.6)
parser.add_argument('--train_dataset', default='/tmp/fonts/small_tx/out/train-00000-of-00001')
parser.add_argument('--validation_dataset', default='/tmp/fonts/small_tx/out/validation-00000-of-00001')
parser.add_argument('--norm_clipping', default=False, type=bool)
parser.add_argument('--use_salt_pepper_noise', default=False, type=bool)
parser.add_argument('--loss_type', choices=['softmax', 'hinge'], default='softmax')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs
args = parser.parse_args()

default_graph = tf.Graph()

train_dataset = TFImageReader(args.train_dataset, args.batch_size,
                              unlimited=True)
val_dataset = TFImageReader(args.validation_dataset, args.batch_size, unlimited=True)


def svm_loss(labels, logits):
    # convert one hot to dense
    y = tf.cast(tf.argmax(labels, axis=1), tf.int32)
    nrows = tf.shape(logits)[0]
    correct_score_indices = tf.stack([tf.range(start=0, limit=nrows), y], axis=1)
    correct_class_scores = tf.gather_nd(logits, correct_score_indices)
    margins = tf.maximum(0.0, logits - tf.reshape(correct_class_scores, [nrows, 1]) + 1.0)
    return tf.reduce_sum(margins - tf.cast(labels, tf.float32)) / tf.cast(nrows, tf.float32)

# For tf.variable_scope vs tf.name_scope,
#  see
#  https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow
with default_graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 1200], name='train_images')
    y_ = tf.placeholder(tf.float32, shape=[None, 37], name='train_labels')

    is_training = tf.placeholder_with_default(False, shape=None, name='is_training')
    with tf.name_scope('reshape'):
        x_input = tf.reshape(x, [-1, 40, 30, 1], name='reshaped_images')

        noise_pepper = tf.constant(salt_and_pepper((40, 30), zero_prob=0.02),
                                   shape=[1, 40, 30, 1], name='pepper_noise', dtype=tf.float32)

        noise_salt = 1 - tf.constant(salt_and_pepper((40, 30), zero_prob=0.98),
                                     shape=[1, 40, 30, 1], name='salt_noise')


        def salt_pepper_noise(data):
            return 255 - (255 - data * salt_and_pepper((1, 40, 30, 1),
                                                       zero_prob=0.01)) * salt_and_pepper((1, 40, 30, 1),
                                                                                          zero_prob=0.01)


        x_noise_input = tf.cond(is_training, lambda: salt_pepper_noise(x_input), lambda: x_input)

        x_image = tf.pad(x_noise_input, [[0, 0], [0, 0], [1, 1], [0, 0]])

    conv_keep_prob = tf.placeholder_with_default(1.0, name='conv_keep_prob', shape=())
    max_norm = tf.constant(4.0)

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 36])
        b_conv1 = bias_variable([36])
        h_conv1 = tf.nn.dropout(tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1),
                                conv_keep_prob, noise_shape=[1, 1, 36])
        tf.summary.histogram("conv1_out", tf.reshape(h_conv1, [-1]))

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 36, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.dropout(tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2),
                                conv_keep_prob, noise_shape=[1, 1, 32])
        tf.summary.histogram("conv2_out", tf.reshape(h_conv2, [-1]))

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 32, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.dropout(tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3), conv_keep_prob, noise_shape=[1, 1, 64])
        tf.summary.histogram("conv3_out", tf.reshape(h_conv3, [-1]))

    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([5 * 4 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool3_flat = tf.reshape(h_pool3, [-1, 5 * 4 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        tf.summary.histogram("fc1_out", tf.reshape(h_fc1, [-1]))

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder_with_default(1.0, name='keep_prob', shape=())
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('readout'):
        W_fc2 = weight_variable([1024, 37])
        b_fc2 = bias_variable([37])

        y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name='y_conv')

    if args.loss_type == 'softmax':
        print("Minimizing softmax loss")
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    else:
        print("Minimizing hinge loss")
        loss = svm_loss(labels=y_, logits=y_conv)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    training_step = tf.train.AdamOptimizer(1e-4).minimize(loss,
                                                          global_step=global_step)
    #    if args.norm_clipping:
    #        with tf.control_dependencies([training_step]):
    #            norm_clipping_step = [
    #                W_conv3.assign(tf.clip_by_norm(W_conv3.read_value(), 4.0, axes=[1, 2])),
    #                W_conv1.assign(tf.clip_by_norm(W_conv1.read_value(), 4.0, axes=[1, 2])),
    #                W_conv2.assign(tf.clip_by_norm(W_conv2.read_value(), 4.0, axes=[1, 2])),
    #                W_fc1.assign(tf.clip_by_norm(W_fc1.read_value(), 4.0, axes=[1])),
    #                W_fc2.assign(tf.clip_by_norm(W_fc2.read_value(), 4.0, axes=[1]))
    #            ]

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='compare_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    tf.summary.scalar(name='accuracy', tensor=accuracy)
    tf.summary.scalar(name='cross_entropy_loss', tensor=loss)
    tf.summary.image(tensor=tf.depth_to_space(tf.transpose(W_conv1, [2, 0, 1, 3]), data_format="NHWC", block_size=6),
                     name='W_conv1')
    summary_writer_train = tf.summary.FileWriter(os.path.join(args.logdir, 'train'), graph=default_graph)
    summary_writer_val = tf.summary.FileWriter(os.path.join(args.logdir, 'validation'), graph=default_graph)
    merged_summaries = tf.summary.merge_all()


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


def run_session():
    with tf.Session(graph=default_graph) as sess:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)
        else:
            sess.run(tf.global_variables_initializer())

        train_dataset_iterator = iter(train_dataset)
        validation_dataset_iterator = iter(val_dataset)
        if args.save_model_dir:
            export_saved_model(args.save_model_dir, session=sess, as_text=True)

        for i in range(args.num_training_steps):
            images, labels = next(train_dataset_iterator)

            summaries, train_step_result, step_id = sess.run([merged_summaries, training_step, global_step],
                                                             feed_dict={x: images, y_: labels,
                                                                        keep_prob: args.dropout_keep_ratio,
                                                                        conv_keep_prob: args.conv_dropout_keep_ratio,
                                                                        is_training: args.use_salt_pepper_noise})
            summary_writer_train.add_summary(summaries, step_id)

            validation_images, validation_labels = next(validation_dataset_iterator)
            summaries, accuracy_val = sess.run([merged_summaries, accuracy],
                                               feed_dict={x: 255 - validation_images, y_: validation_labels})
            summary_writer_val.add_summary(summaries, step_id)

            if i % args.checkpoint_every == 0:
                feed_dict_eval = {
                    x: images, y_: labels, keep_prob: 1.0}
                train_accuracy = sess.run(accuracy, feed_dict=feed_dict_eval)
                if args.checkpoint_dir:
                    saver.save(sess, os.path.join(args.checkpoint_dir,
                                                  CHECKPOINT_FILE_NAME),
                               global_step=step_id,
                               write_meta_graph=False)
                print("step %d, accuracy=%f, global_step=%d" % (i, train_accuracy, step_id))


run_session()
