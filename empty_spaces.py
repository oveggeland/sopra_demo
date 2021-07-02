import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from clustering import DBSCAN_cluster
from visualizer import create_heat_map
from image_utils import mask_image

CORNERS = np.array([
    [1000, 400],
    [1600, 200],
    [2100, 1536],
    [300, 1536]
])

VIDEO_NUM = 2
IMG_NUM = 1
N_IMAGES = 10000
IMAGE_INTERVAL = 10

"""
Analyze image stream to find areas with little activity over time.

Params:
    vid_num - ID of video to analyse
    img_num - Starting image number of analysis
    n_images - Number of images to include in analysis
    image_interval - Number of images skipped in analysis

Returns:
    n_regions - Number of regions with low activity
"""
def find_empty_spaces(vid_num=VIDEO_NUM, img_num=IMG_NUM, n_images=N_IMAGES, image_interval=IMAGE_INTERVAL):
    first_img = cv.imread(f"data/video{vid_num}/img{img_num}.jpg", cv.IMREAD_GRAYSCALE).astype('float')
    cum_array = np.zeros(first_img.shape)

    # Loop over each image and add pixel intensities to a cumulative array
    for img in range(1, n_images+1, image_interval):
        print(f"Image {img}")
        dir = f"data/video{vid_num}/img{img}.jpg"
        frame = cv.imread(dir, cv.IMREAD_GRAYSCALE)
        frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, 0)
        cum_array += frame.astype('float')

    cum_array /= (n_images/image_interval)
    cum_array = cum_array.astype(np.uint8)

    # Mask out areas of interest
    cum_array = mask_image(cum_array, CORNERS)


    # Transform to rectangular grid
    #heat_map = create_heat_map(cum_array, CORNERS)


    cum_array = cv.resize(cum_array, [960, 540])
    cv.imshow(f"Cumulative distribution over {n_images} frames", cum_array)

    plt.figure()
    plt.imshow(cum_array)
    plt.savefig("figs/empty_areas.eps", format="eps")

    # Try clustering black areas to see if any large regions are inhabitable
    black_regions = np.where(cum_array < 40, 1, 0).astype(np.uint8)
    clustered_img, n_regions = DBSCAN_cluster(black_regions, min_size=1000)
    return n_regions
