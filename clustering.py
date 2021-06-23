import cv2 as cv
import numpy as np
import time


def kmeans_cluster(img, k=10):
    # Create clusters of an black/white image
    white_pixels = np.where(img)    # (x-coordinates, y-coordinates)
    coordinates = np.float32(np.column_stack((white_pixels[0], white_pixels[1])))

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)

    _, _, centers = cv.kmeans(coordinates, k, None, criteria, 1, cv.KMEANS_PP_CENTERS)

    return centers.astype('int')
