"""Utilities for image data augmentation via image transforms"""

import numpy as np
import cv2
import imutils
import string
import os.path
from PIL import Image, ImageFont, ImageDraw
import glob

all_chars_list = [c for c in string.ascii_uppercase] + [c for c in string.digits]


def gen_all_char_images(image_dir):
    f = ImageFont.truetype('Arial.ttf', 35)
    for c in all_chars_list:
        image = Image.new(color=255, mode='L', size=(30, 40))
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), c, font=f)
        file_path = os.path.join(image_dir, c + '.png')
        image.save(file_path, 'png')


def rotate_image(img, deg):
    img_rot = imutils.rotate_bound(img, deg)
    white_mask = imutils.rotate_bound(np.ones_like(img) * 255, deg)
    # Invert mask
    white_mask = 255 - white_mask
    return img_rot + white_mask


def translate_image(img, tx_matrix):
    img_tx = cv2.warpAffine(img, tx_matrix, dsize=(img.shape[1], img.shape[0]))
    white_mask = cv2.warpAffine(np.ones_like(img) * 255, tx_matrix, dsize=(img.shape[1], img.shape[0]))
    white_mask = 255 - white_mask
    return img_tx + white_mask


def get_warp_matrix(img, warp_spec):
    orig = np.float32(pt_orig * img.shape)
    dest = np.float32(warp_spec * img.shape)
    return cv2.getPerspectiveTransform(orig, dest)


def warp_perspective(img, warp_matrix):
    img_warp = cv2.warpPerspective(img, warp_matrix, (img.shape[1], img.shape[0]))
    white_mask = cv2.warpPerspective(np.ones_like(img) * 255, warp_matrix, dsize=(img.shape[1], img.shape[0]))
    white_mask = 255 - white_mask
    return img_warp + white_mask


# points in clockwise starting topleft
pt_orig = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])

top_shrink = np.float32([[0.1, 0], [0.9, 0], [1, 1], [0, 1]])
bottom_shrink = np.float32([[0, 0], [1, 0], [0.9, 1], [0.1, 1]])
left_shrink = np.float32([[0, 0.1], [1, 0], [1, 1], [0, 0.9]])
right_shrink = np.float32([[0, 0], [1, 0.1], [1, 0.9], [0, 1]])

rl_diag_shrink = np.float32([[0, 0], [0.9, 0.1], [1, 1], [0.1, 0.9]])
lr_diag_shrink = np.float32([[0.1, 0.1], [1, 0], [0.9, 0.9], [0, 1]])

warp_specs = [
    top_shrink,
    bottom_shrink,
    left_shrink,
    right_shrink,
    rl_diag_shrink,
    lr_diag_shrink,
]

translate_matrices = []
for i in range(-5, 6, 2):
    for j in range(-5, 6, 2):
        translate_matrices.append(np.float32(np.array([1, 0, i, 0, 1, j]).reshape(2, 3)))

rotation_deg = [deg for deg in range(-10, 11, 2)]

base_images_dir = '/tmp/fonts/arial_35'

files_list = []

np.random.seed(0)

TRAIN = 0.6
VAL = 0.8
TEST = 1.0

train_dir = '/tmp/fonts/train'
val_dir = '/tmp/fonts/val'
test_dir = '/tmp/fonts/test'

labels_file_path = '/tmp/fonts/labels.txt'


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def get_random_dir():
    p = np.random.sample()
    if p < TRAIN:
        return train_dir
    elif p < VAL:
        return val_dir
    return test_dir


def save_image(label, filename, image):
    outdir = os.path.join(get_random_dir(), label)
    ensure_dir(outdir)
    path = os.path.join(outdir, filename)
    cv2.imwrite(path, image)


labels = set({})


def main(unused_argv):
    ensure_dir(train_dir)
    ensure_dir(val_dir)
    ensure_dir(test_dir)

    for f in glob.glob(os.path.join(base_images_dir, '*.png')):
        name, ext = os.path.basename(f).split('.')
        label = name
        labels.add(label)
        filename = name + '_orig.jpg'
        orig = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        save_image(name, filename, orig)

        for i in range(len(translate_matrices)):
            for j in range(len(rotation_deg)):
                for k in range(len(warp_specs)):
                    img = translate_image(orig, translate_matrices[i])
                    img = cv2.resize(img, dsize=(orig.shape[1], orig.shape[0]))
                    img = rotate_image(img, rotation_deg[j])
                    img = cv2.resize(img, dsize=(orig.shape[1], orig.shape[0]))
                    img = warp_perspective(img, get_warp_matrix(img, warp_specs[k]))
                    img = cv2.resize(img, dsize=(orig.shape[1], orig.shape[0]))

                    filename = "%s_%d_%d_%d.jpeg" % (name, i, j, k)
                    save_image(label, filename, img)

    labels_file = open(labels_file_path, 'w')
    for key in labels:
        labels_file.write(key)
        labels_file.write('\n')

    labels_file.close()


if __name__ == '__main__':
    main(None)
