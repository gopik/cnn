import tensorflow as tf
import os.path
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import glob
import cv2

LOG_DIR = '/tmp/embedding/logs'

# Create randomly initialized embedding weights which will be trained.
vocabulary_size = 40
embedding_size = 64


# # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
# config = projector.ProjectorConfig()
#
# # You can add multiple embeddings. Here we add only one.
# embedding = config.embeddings.add()
# embedding.tensor_name = embedding_var.name
# # Link this tensor to its metadata file (e.g. labels).
# embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
#
# # Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

images = np.zeros(shape=(vocabulary_size, embedding_size))

sprite = np.zeros(64*64*32*32)

i = 0

metadata = open( os.path.join(LOG_DIR, 'metadata.tsv'), 'w')
for f in glob.glob('/Users/gopik/Downloads/chars/resized/*.jpg'):
    if i == 40:
        break
    metadata.write('%s\n' % f)
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sprite[i * 32 * 32 : (i + 1) * 32 * 32] = cv2.resize(img, dsize=(32, 32)).reshape(32 * 32)
    images[i * 64: (i + 1) * 64] = cv2.resize(img, dsize=(8, 8)).reshape(64)
    i += 1

metadata.close()
# sprite = sprite.reshape(2048, 2048)
# cv2.imwrite('/tmp/embedding/logs/sprite.png', sprite)
# embedding.sprite.image_path = '/tmp/embedding/logs/sprite.png'
# embedding.sprite.single_image_dim.extend([2048, 2048])

embedding = tf.Variable(np.zeros(shape=(40, 64)), name='embedding')
# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
#projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()

with tf.Session(graph=tf.get_default_graph()) as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, '/tmp/embedding/logs/ckpt')

summary_writer.close()