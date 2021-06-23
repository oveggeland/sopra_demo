import cv2 as cv
import numpy as np
import time

def cluster(img, K=10, draw=False):
    # Create clusters of an black/white image
    white_pixels = np.where(img)
    coordinates = np.float32(np.column_stack((white_pixels[0], white_pixels[1])))

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)

    retVal, best_labels, centers = cv.kmeans(coordinates, K, None,\
                                             criteria, 1, cv.KMEANS_PP_CENTERS)

    return centers.astype('int')
