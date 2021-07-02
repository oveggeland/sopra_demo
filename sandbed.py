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


"""
Counts the amount of chickens over a specified time period and creates a graph of the result

Params:
    strategy - Strategy for counting chickens (clusters, area, yolo)
    video_num - ID of video to analyse
    img_num - First image to start analysing
    n_images - Number of images to analyse
    image_interval - Number of images to skip between each analysis
    frame_time - Time between each image frame
"""
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


"""
Counts the number of chickens in a frame based on the white to black area ratio. 

Params:
    video_num - ID of video to analyse
    img_num - ID of image to analyse

Returns:
    n_chickens - Number of chickens estimated to be in img
"""
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


"""
Counts the number of chickens in a frame based the amount of reasonably sized white clusters found. 

Params:
    video_num - ID of video to analyse
    img_num - ID of image to analyse

Returns:
    n_chickens - Number of chickens estimated to be in img
"""
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
    cluster_img, n_chickens = DBSCAN_cluster(threshold_img)

    # Save figure with result
    img = cv.resize(img, (cluster_img.shape[1], cluster_img.shape[0]))
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f'Chicken detections by DBSCAN clustering. {n_chickens} chickens found')
    axs[0].imshow(img)
    axs[1].imshow(cluster_img)
    plt.tight_layout()
    plt.savefig("figs/db_cluster.eps")
    plt.savefig("figs/db_cluster.png")

    cv.imshow("img", img)
    cv.imshow("cluster_img", cluster_img)

    key = cv.waitKey()
    if key & 0xFF == ord('q'):
        exit()

    return n_chickens


if __name__ == "__main__":
    chickens_over_time()
