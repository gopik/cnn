import argparse
from scipy import ndimage
from recognizer import Recognizer
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Train CNN for MNIST')
    parser.add_argument('--save_model_dir', help='Path to save exported model. Model will be exported only if provided')
    parser.add_argument('--input_image')
    args = parser.parse_args()

    image = ndimage.imread(args.input_image, flatten=True)
    with Recognizer(args.save_model_dir) as recognizer:
        print(recognizer.predict(image[np.newaxis, :, :, np.newaxis]))
        print(np.argmax(recognizer.predict(image[np.newaxis, :, :, np.newaxis])))


if __name__ == '__main__':
    main()
