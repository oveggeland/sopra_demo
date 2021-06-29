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


def dlt(xy, XY):
    n_points = xy.shape[1]
    assert n_points == 4, "DLT should get four point correspondence to work"
    A = np.zeros((8, 9))
    for i in range(n_points):
        A[2*i:2*(i+1), :] = np.array([
            [XY[0, i], XY[1, i], 1, 0, 0, 0, -XY[0, i]*xy[0, i], -XY[1, i]*xy[0, i], -xy[0, i]],
            [0, 0, 0, XY[0, i], XY[1, i], 1, -XY[0, i]*xy[1, i], -XY[1, i]*xy[1, i], -xy[1, i]]
        ])

    _, _, V = np.linalg.svd(A)
    V = V.T
    h = V[:, -1]  # Last column of V
    H = np.reshape(h, [3, 3])

    return H

