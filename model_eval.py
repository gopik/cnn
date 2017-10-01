import numpy as np
import pandas as pd
import os.path
import glob
from recognizer import Recognizer
import cv2
import string
import argparse

parser = argparse.ArgumentParser(
    description='Evaluate model accuracy on test data and saves per cat accuracy in model dir')
parser.add_argument('--save_model_dir', help='Path to save exported model.')
args = parser.parse_args()

chars = [None] + list(string.digits) + list(string.ascii_uppercase)


def predict(r, path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img[np.newaxis, :, :, np.newaxis]
    return chars[np.argmax(r.predict(img))]


def main():
    rec = glob.glob('/Users/gopik/Downloads/container/outputs/**/recognized*', recursive=True)
    df = pd.DataFrame({'file_path': rec})
    df = df.assign(cat=lambda d: d['file_path'].apply(lambda fp: os.path.basename(fp).split('_')[1]))
    r = Recognizer(args.save_model_dir)

    df_prediction = df.assign(prediction=lambda d: d['file_path'].apply(lambda path: predict(r, path)))
    df_prediction['acc'] = df_prediction['cat'] == df_prediction['prediction']
    df_acc = df_prediction.groupby(['cat', 'acc']).count().unstack()['prediction'].fillna(0)
    df_acc['total'] = df_acc[False] + df_acc[True]
    df_acc['percent'] = df_acc[True] / df_acc['total']
    df_acc.to_csv(os.path.join(args.save_model_dir, 'acc.csv'))
    print(np.sum(df_acc[True])/np.sum(df_acc['total']))


if __name__ == '__main__':
    main()
