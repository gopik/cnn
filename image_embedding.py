import tensorflow as tf
import os.path
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import glob
import cv2
import argparse
from recognizer import Recognizer


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='log dir to store checkpoints and metadata')
parser.add_argument('--max_images', type=int, help='number of images to load', default=900)
parser.add_argument('--images_dir', help='directory containing images')
parser.add_argument('--save_model_dir', help='Saved model directory')

args = parser.parse_args()

# Create randomly initialized embedding weights which will be trained.
num_images = args.max_images
sprite_image_height = 16
sprite_image_width = 16

image_height = 28
image_width = 28


def load_images(images_dir, max_images):
    """Loads images from files in directory and returns a np array of shape (n, h, w)

    Args:
        images_dir: Directory to load images from.
        max_images: Maximum number of images to load.

    Returns:
        (images(shape=[n, h, w]), file_list([string])"""
    i = 0
    img_list = []
    file_list = []
    print(max_images)
    for f in glob.glob(os.path.join(images_dir, '*.jpg')):
        if i == max_images:
            break
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(28, 28))
        # Extend a new axis along which the list of images will be concatenated.
        img = img[np.newaxis, ...]
        img_list.append(img)
        file_list.append(os.path.basename(f))
        i += 1
        print(i)
    print(len(img_list))
    return np.vstack(img_list), file_list


def recognize_images(images):
    """Given a tensor of images (N, h, w), returns predictions as (N, num_classes)"""
    with Recognizer(model_dir=args.save_model_dir) as recognizer:
        return recognizer.predict(images/255.0)


def create_sprite_image(images, height, width):
    """Creates sprite image from input images.

    Full sprite image is a square. The individual small images need not be square. The images are stored in the sprite
    in a row major order.

    :param images: numpy array of shape (num_images, actual_image_height, actual_image_width).
    :param height: height of individual image in the sprite.
    :param width: width of individual image in the sprite.
    :return: Numpy array representing sprite image."""

    num_images_per_row = int(np.ceil(np.sqrt(images.shape[0])))
    sprite = np.zeros(shape=(num_images_per_row * height, num_images_per_row * width))
    nrows = num_images_per_row

    for idx in range(images.shape[0]):
        img = images[idx]
        i, j = divmod(idx, nrows)
        sprite_img = cv2.resize(img, dsize=(height, width))
        pixel_row_start, pixel_col_start = i * height, j * width
        pixel_row_end, pixel_col_end = (i + 1) * height, (j + 1) * width
        sprite[pixel_row_start:pixel_row_end, pixel_col_start:pixel_col_end] = sprite_img

    return sprite


def main():
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()

    images, file_list = load_images(args.images_dir, args.max_images)
    images_tensor = recognize_images(images)
    graph = tf.Graph()
    with graph.as_default():
        embedding_var = tf.Variable(images_tensor, dtype=tf.float32, name='embedding')
    embedding.tensor_name = embedding_var.name

    metadata_path = os.path.join(args.logdir, 'metadata.tsv')
    embedding.metadata_path = metadata_path
    with open(metadata_path, 'w') as metadata:
        for file_name in file_list:
            metadata.write(file_name)
            metadata.write('\n')

    sprite_image = create_sprite_image(images, sprite_image_height, sprite_image_width)
    sprite_image_path = os.path.join(args.logdir, 'sprite.png')
    cv2.imwrite(sprite_image_path, sprite_image)

    embedding.sprite.image_path = sprite_image_path
    embedding.sprite.single_image_dim.extend([sprite_image_height, sprite_image_width])

    summary_writer = tf.summary.FileWriter(args.logdir, graph=tf.get_default_graph())

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)

    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, '/tmp/embedding/logs/ckpt')

    summary_writer.close()


if __name__ == '__main__':
    main()
