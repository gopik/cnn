import tensorflow as tf
import os.path
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import glob
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='log dir to store checkpoints and metadata')
parser.add_argument('--max_images', type=int, help='number of images to load', default=900)
parser.add_argument('--images_dir', help='directory containing images')

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
        print(i, '==', max_images)
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


def create_sprite_image(images, height, width):
    """Creates sprite image from input images.

    Full sprite image is a square. The individual small images need not be square. The images are stored in the sprite
    in a row major order.

    Args:
        images: numpy array of shape (num_images, actual_image_height, actual_image_width).
        height: height of individual image in the sprite.
        width: width of individual image in the sprite.

    Returns: numpy array representing the sprite image."""

    num_images_per_row = int(np.ceil(np.sqrt(images.shape[0])))
    print(images.shape[0])
    print(num_images_per_row)
    sprite = np.zeros(shape=(num_images_per_row * height, num_images_per_row * width))
    nrows = num_images_per_row
    ncols = num_images_per_row

    for i in range(nrows):
        for j in range(ncols):
            image_idx = i * ncols + j
            if image_idx == images.shape[0]:
                break
            img = images[image_idx]
            sprite_img = cv2.resize(img, dsize=(height, width))
            pixel_row_start, pixel_col_start = i * height, j * width
            pixel_row_end, pixel_col_end = (i + 1) * height, (j + 1) * width
            sprite[pixel_row_start:pixel_row_end, pixel_col_start:pixel_col_end] = sprite_img

    return sprite


def main():
    images, file_list = load_images(args.images_dir, args.max_images)
    images_tensor = images.reshape(images.shape[0], -1)

    sprite_image = create_sprite_image(images, sprite_image_height, sprite_image_width)
    sprite_image_path = os.path.join(args.logdir, 'sprite.png')
    cv2.imwrite(sprite_image_path, sprite_image)

    metadata_path = os.path.join(args.logdir, 'metadata.tsv')
    with open(metadata_path, 'w') as metadata:
        for file_name in file_list:
            metadata.write(file_name)
            metadata.write('\n')

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding_var = tf.Variable(images_tensor, dtype=tf.float32, name='embedding')
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata_path
    summary_writer = tf.summary.FileWriter(args.logdir, graph=tf.get_default_graph())
    embedding.sprite.image_path = sprite_image_path
    embedding.sprite.single_image_dim.extend([sprite_image_height, sprite_image_width])

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)

    saver = tf.train.Saver()

    with tf.Session(graph=tf.get_default_graph()) as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, '/tmp/embedding/logs/ckpt')

    summary_writer.close()


if __name__ == '__main__':
    main()
