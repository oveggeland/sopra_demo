import cv2 as cv
import numpy as np

from image_utils import mask_image, crop_image
from clustering import kmeans_cluster, DBSCAN_cluster, affinity_cluster
from visualizer import draw_circles

# Defining bed corners
OUTER_BED_CORNERS = np.array([
        [930, 400],
        [930, 910],
        [1600, 900],
        [1550, 400]
    ], np.int32)

INNER_BED_CORNERS = np.array([
        [1010, 460],
        [1010, 840],
        [1530, 830],
        [1500, 460]
    ], np.int32)


def find_chickens(vid_num, img_num):
    dir = f'data/video{vid_num}/img{img_num}.jpg'
    img = cv.imread(dir)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    masked_img = mask_image(img, INNER_BED_CORNERS)

    img = crop_image(img, OUTER_BED_CORNERS)
    masked_img = crop_image(masked_img, OUTER_BED_CORNERS)

    # Smooth images
    blurred_img = cv.bilateralFilter(masked_img, 20, 20, 20)

    # Apply white/black thresholding
    threshold_img = cv.adaptiveThreshold(blurred_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, -5)

    # Show different results
    #cv.imshow('Frame', img)
    cv.imshow('Blurred', blurred_img)
    cv.imshow('Threshold', threshold_img)

    # Cluster white areas to detect chickens
    centers = []
    #centers = kmeans_cluster(threshold_img, k=20)
    #centers = affinity_cluster(threshold_img)
    img, n_clusters = DBSCAN_cluster(threshold_img)

    key = cv.waitKey()
    if key & 0xFF == ord('q'):
        exit()

    return n_clusters