import os
import os.path
import re

import cv2
import numpy as np
import glob
import utils


def recursive_find_files(root_dir, pattern):
    """Returns a list of files matching the pattern starting from root_dir"""
    files_list = []
    matcher = re.compile(pattern)
    for dirname, subdirs, files in os.walk(root_dir):
        files_list += filter(lambda f: matcher.match(f),
                             map(lambda filename: os.path.join(os.path.abspath(dirname), filename), files))
    return files_list


class Frame(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.orig = os.path.join(dirname, 'orginal.jpg')
        self.rec = glob.glob(os.path.join(dirname, 'chars/**/rec*.jpg'), recursive=True)
        self.original = cv2.imread(self.orig, cv2.IMREAD_GRAYSCALE)

    def get_orig_crop(self):
        ex = {}
        for file in self.rec:
            _, cat, x, y, w, h = os.path.basename(file).split('.')[0].split('_')
            x, y, w, h = map(int, [x, y, w, h])
            orig_crop = self.original[y:y + h, x:x + w]
            crop = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            _, img = cv2.threshold(orig_crop, 36, 255, cv2.THRESH_BINARY_INVo)
            img_pad = np.pad(img, utils.get_padding(h, w, 40, 30), mode='constant', constant_values=255)
            img_pad_resize = cv2.resize(img_pad, (30, 40), interpolation=cv2.INTER_CUBIC)

            ex[file] = (cat, orig_crop, img_pad_resize)
        return ex

    def save_images(self, outdir, result):
        for key, (cat, orig, img) in result.items():
            target_dir = os.path.join(outdir, cat)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            f = os.path.basename(self.dirname) + '_' + os.path.basename(key).split('.')[0] + ".jpeg"
            cv2.imwrite(os.path.join(target_dir, f), img)
