import numpy as np
import pandas as pd
import os.path
import glob
from recognizer import Recognizer
import cv2
import string

chars = [None] + list(string.digits) + list(string.ascii_uppercase)


def predict(r, path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img[np.newaxis, :, :, np.newaxis]
    prediction = r.predict(img)
    return chars[np.argmax(prediction)]


def main(unused_args):
    rec = glob.glob('/Users/gopik/Downloads/container/outputs/**/recognized*', recursive=True)
    df = pd.DataFrame({'file_path': rec})
    df = df.assign(cat=lambda d: d['file_path'].apply(lambda fp: os.path.basename(fp).split('_')[1]))
    r = Recognizer('fonts/saved_model_1')

    df_prediction = df.assign(prediction=lambda d: d['file_path'].apply(lambda path: predict(r, path)))
    df_prediction['acc'] = df_prediction['cat'] == df_prediction['prediction']
    df_acc = df_prediction.groupby(['cat', 'acc']).count().unstack()['prediction'].fillna(0)
    df_acc['total'] = df_acc[False] + df_acc[True]
    df_acc['percent'] = df_acc[True]/df_acc['total']
    print(np.mean(df_acc['total'] * df_acc['percent']))


if __name__ == '__main__':
    main(None)
