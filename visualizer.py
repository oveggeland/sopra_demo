import cv2 as cv
import numpy as np


def draw_circles(img, centers, radius=5, linesize=3):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for i in range(centers.shape[0]):
        center = (centers[i, 1], centers[i, 0])
        cv.circle(img, center, radius, (0, 255, 0), linesize)
    cv.imshow('Circles', img)
