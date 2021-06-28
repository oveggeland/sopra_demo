import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from image_utils import mask_image, crop_image
from clustering import DBSCAN_cluster

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

VIDEO_NUM = 1
IMG_NUM = 1
N_IMAGES = 50
IMAGE_INTERVAL = 100
FRAME_TIME = 1/19.72


def chickens_over_time(strategy='clusters', video_num=VIDEO_NUM, img_num=IMG_NUM, n_images=N_IMAGES,\
                       image_interval=IMAGE_INTERVAL, frame_time=FRAME_TIME):
    chicken_counter = []
    time = []
    for i in range(n_images):
        if strategy=='clusters':
            n_chickens = find_chickens_by_clusters(video_num, img_num+image_interval*i)
        else:
            n_chickens = find_chickens_by_area(video_num, img_num+image_interval*i)

        chicken_counter.append(n_chickens)
        time.append((img_num+image_interval*i)*frame_time)

    plt.plot(time, chicken_counter)
    plt.show()


def find_chickens_by_area(vid_num, img_num):
    dir = f'data/video{vid_num}/img{img_num}.jpg'
    img = cv.imread(dir)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = crop_image(img, OUTER_BED_CORNERS)

    # Blur and threshold image
    blurred_img = cv.bilateralFilter(img, 20, 20, 20)
    threshold_img = cv.adaptiveThreshold(blurred_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, -10)

    total_area = img.size
    white_area = np.where(threshold_img)[0].size
    white_ratio = white_area/total_area

    average_chicken_size = 4000
    n_chickens = white_area // average_chicken_size
    #cv.imshow(f'Number of chickens by area approximation: {n_chickens}', img)
    #cv.waitKey()
    #cv.destroyAllWindows()

    return n_chickens


def find_chickens_by_clusters(vid_num, img_num):
    dir = f'data/video{vid_num}/img{img_num}.jpg'
    img = cv.imread(dir)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    masked_img = mask_image(img, INNER_BED_CORNERS)

    img = crop_image(img, OUTER_BED_CORNERS)
    masked_img = crop_image(masked_img, OUTER_BED_CORNERS)

    # Smooth images
    blurred_img = cv.bilateralFilter(masked_img, 20, 20, 20)

    # Apply white/black thresholding
    threshold_img = cv.adaptiveThreshold(blurred_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, -10)

    # Show different results
    #cv.imshow('Frame', img)
    #cv.imshow('Blurred', blurred_img)
    #cv.imshow('Threshold', threshold_img)

    # Cluster white areas to detect chickens
    img, n_clusters = DBSCAN_cluster(threshold_img)

    key = cv.waitKey()
    if key & 0xFF == ord('q'):
        exit()

    return n_clusters
