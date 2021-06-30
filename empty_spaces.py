import cv2 as cv
import numpy as np

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
N_IMAGES = 10
IMAGE_INTERVAL = 10


def find_empty_spaces(vid_num=VIDEO_NUM, img_num=IMG_NUM, n_images=N_IMAGES, image_interval=IMAGE_INTERVAL):
    first_img = cv.imread(f"data/video{vid_num}/img{img_num}.jpg", cv.IMREAD_GRAYSCALE).astype('float')
    cum_array = np.zeros(first_img.shape)
    for img in range(1, n_images+1, image_interval):
        print(f"Doing stuff with image {img}")
        dir = f"data/video{vid_num}/img{img}.jpg"
        frame = cv.imread(dir, cv.IMREAD_GRAYSCALE)
        frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, 0)
        cum_array += frame.astype('float')

    cum_array /= (n_images/image_interval)
    cum_array = cum_array.astype(np.uint8)

    # Mask out area of interest defined by CORNERS
    cum_array = mask_image(cum_array, CORNERS)


    #Reshape and show on screen!
    create_heat_map(cum_array, CORNERS)
    cum_array = cv.resize(cum_array, [960, 540])
    cv.imshow(f"Cumulative distribution over {n_images} frames", cum_array)

    # Try clustering black areas to see if any large regions are inhabitable
    black_regions = np.where(cum_array < 40, 1, 0).astype(np.uint8)
    n_regions = DBSCAN_cluster(black_regions)

    # Press any button to move on. Pressing 'q' exits the entire script
    key = cv.waitKey()
    if key & 0xFF == ord('q'):
        exit()
