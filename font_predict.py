import argparse
from scipy import ndimage
from recognizer import Recognizer
import numpy as np
import glob

classes = np.array(['Unknown', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
           'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])


def main():
    parser = argparse.ArgumentParser(description='Arial font character predictor')
    parser.add_argument('--save_model_dir', help='Path to save exported model.')
    parser.add_argument('--input_image')
    args = parser.parse_args()

    with Recognizer(args.save_model_dir) as recognizer:
        for f in glob.glob('/tmp/fonts/small_tx/test/0/*.jpeg'):
            image = ndimage.imread(f, flatten=True)
            reshaped_image = image[np.newaxis, :, :, np.newaxis]
            prediction = recognizer.predict(reshaped_image)
            indices = np.argsort(-prediction)
            print("%s %s" % (f, classes[indices]))


if __name__ == '__main__':
    main()
