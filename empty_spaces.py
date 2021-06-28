import cv2 as cv
import numpy as np

from clustering import DBSCAN_cluster

CORNERS = np.array([
    [1000, 400],
    [1600, 200],
    [2100, 1540],
    [300, 1540]
])

VIDEO_NUM = 2
IMG_NUM = 1
N_IMAGES = 10000
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
    img_mask = np.zeros(cum_array.shape)
    img_mask = cv.fillPoly(img_mask, [CORNERS], 1)
    cum_array = np.where(img_mask == 1, cum_array, 255)


    #Reshape and show on screen!
    cum_array = cv.resize(cum_array, [960, 540])
    cv.imshow(f"Cumulative distribution over {n_images} frames", cum_array)

    # Try clustering black areas to see if any large regions are inhabitable
    black_regions = np.where(cum_array < 40, 1, 0).astype(np.uint8)
    n_regions = DBSCAN_cluster(black_regions)

    # Press any button to move on. Pressing 'q' exits the entire script
    key = cv.waitKey()
    if key & 0xFF == ord('q'):
        exit()
