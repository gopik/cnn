import argparse
from tensorflow.examples.tutorials.mnist import input_data
from recognizer import Recognizer


def main():
    parser = argparse.ArgumentParser(description='Train CNN for MNIST')
    parser.add_argument('--save_model_dir', help='Path to save exported model. Model will be exported only if provided')
    args = parser.parse_args()
    mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

    with Recognizer(args.save_model_dir) as recognizer:
        print(recognizer.predict(mnist.test.images[0]))


if __name__ == '__main__':
    main()
