import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset


class TFImageReader(object):
    """An iterator for TF record based images"""

    def __init__(self, dataset, batch_size, unlimited=False):
        """
        Create a new TFRecord based image reader. Tensorflow ops are added to a new graph.

        :param dataset: Dataset file name (RecordIO containing tf.Example protos)
        :param batch_size: Number of images to return in one read.
        :param unlimited: If true, the iteration will never end.
        """
        self.graph = tf.Graph()
        repeat_count = -1
        if not unlimited:
            repeat_count = 1
        with self.graph.as_default():
            self.dataset = TFRecordDataset(dataset).map(self.get_feature).repeat(repeat_count).shuffle(10000).batch(
                batch_size)

        self.session = tf.Session(graph=self.graph)

    def __iter__(self):
        with self.graph.as_default():
            self.iterator_tensor = self.dataset.make_one_shot_iterator().get_next()
            return self

    def __next__(self):
        try:
            image, label = self.session.run(self.iterator_tensor)
            return image, label
        except tf.errors.OutOfRangeError:
            raise StopIteration()

    def get_feature(self, example_proto):
        parsed_feature = self.parse_function(example_proto)

        with tf.name_scope('decode_jpeg'):
            decoded_image = tf.image.decode_jpeg(parsed_feature['image/encoded'])
            image = tf.reshape(tf.image.rgb_to_grayscale(decoded_image, "rgb_to_grayscale"),
                               shape=[40 * 30])
            label = tf.one_hot(parsed_feature['image/class/label'], depth=37)
        return image, label

    def parse_function(self, example_proto):
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
