"""Utilities for image data augmentation via image transforms"""

import numpy as np
import cv2
import imutils


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
        translate_matrices.append(np.array([1, 0, i, 0, 1, j]).reshape(2, 3))

rotation_deg = [deg for deg in range(-10, 11, 2)]