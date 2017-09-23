import argparse
from scipy import ndimage
from recognizer import Recognizer
import numpy as np

classes = ['Unknown', 'L', 'S', 'H', 'G', '9', 'V', '8', 'Q', '5', 'T', '7', 'C', '6', 'O', 'M', 'P', 'Z', 'N',
           'E', '3', 'X', '4', 'I', 'U', 'D', 'W', 'R', 'J', 'B', 'K', 'Y', 'A', '2', '1', '0', 'F']

def main():
    parser = argparse.ArgumentParser(description='Train CNN for MNIST')
    parser.add_argument('--save_model_dir', help='Path to save exported model. Model will be exported only if provided')
    parser.add_argument('--input_image')
    args = parser.parse_args()

    image = ndimage.imread(args.input_image, flatten=True)
    with Recognizer(args.save_model_dir) as recognizer:
        reshaped_image = image[np.newaxis, :, :, np.newaxis]
        prediction = recognizer.predict(reshaped_image)
        print(prediction)

        class_index = np.argmax(prediction)
        print(class_index)

        print(classes[class_index])


if __name__ == '__main__':
    main()
