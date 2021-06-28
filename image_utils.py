import cv2 as cv
import numpy as np


def mask_image(img, pts):
    img_mask = np.zeros(img.shape)
    cv.fillPoly(img_mask, [pts], 1)
    return img * img_mask.astype(np.uint8)


def crop_image(img, pts):
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    return img[min_y:max_y, min_x:max_x]