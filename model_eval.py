import numpy as np
import pandas as pd
import os.path
import glob
from recognizer import Recognizer
import cv2
import string
import argparse
import utils

parser = argparse.ArgumentParser(
    description='Evaluate model accuracy on test data and saves per cat accuracy in model dir')
parser.add_argument('--save_model_dir', help='Path to save exported model.')
args = parser.parse_args()

chars = [None] + list(string.digits) + list(string.ascii_uppercase)


def predict(r, path):
    #filename, _ = os.path.basename(path).split('.')
    #_, _, x, y, w, h = filename.split('_')
    #x, y, w, h = int(x), int(y), int(w), int(h)
#    print(x, y, w, h)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#    img = cv2.medianBlur(img, 3)
    #img_resize = cv2.resize(img, (w, h), cv2.INTER_NEAREST)
    h, w = img.shape
    img_pad = utils.get_padding(h, w, 40, 30)
    padded_img = np.pad(img, img_pad, mode='constant', constant_values=255)
    img = cv2.resize(padded_img, (30, 40), cv2.INTER_AREA)
    img = img[np.newaxis, :, :, np.newaxis]
    prediction = r.predict(255 - img)
#    print(prediction)
    result = (chars[np.argmax(prediction[:, :11])], chars[11 + np.argmax(prediction[:, 11:])])
    #return chars[np.argmax(prediction)]
    return result


def main():
    sample_df = pd.DataFrame.from_csv('data/samples.csv')
    rec = glob.glob('/home/gopik/github/cnn/data/lot1/outputs/**/recognized*.jpg', recursive=True)
    df = pd.DataFrame({'file_path': rec})
    df = df.assign(cat=lambda d: d['file_path'].apply(lambda fp: os.path.basename(fp).split('_')[1]))
    r = Recognizer(args.save_model_dir)

    df_prediction = df.assign(prediction=lambda d: d['file_path'].apply(lambda path: predict(r, path)))
    df_prediction['acc'] = df_prediction.apply(lambda r:
                                               r['cat'] in r['prediction'], axis=1)
    df_prediction.to_csv('model_output.csv')
    df_acc = df_prediction.groupby(['cat', 'acc']).count().unstack()['prediction'].fillna(0)
    print(df_acc)
    df_acc['total'] = df_acc[False] + df_acc[True]
    df_acc['percent'] = df_acc[True] / df_acc['total']
    df_acc.to_csv(os.path.join(args.save_model_dir, 'acc.csv'))
    print(np.sum(df_acc[True])/np.sum(df_acc['total']))


if __name__ == '__main__':
    main()
