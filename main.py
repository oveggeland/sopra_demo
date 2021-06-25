import cv2 as cv
import numpy as np

from clustering import kmeans_cluster
from visualizer import draw_circles

# Specify what image/frame to analyze
VIDEO_NUM = 2
IMG_NUM = 1
N_IMAGES = 10000
IMAGE_INTERVAL = 20

CORNERS = np.array([
    [1000, 400],
    [1600, 200],
    [2100, 1540],
    [300, 1540]
])

def main():
    #read_video()        # Read video and save individual images
    find_empty_spaces()


#TODO: Fjern forskjellen i lysstyrke
def find_empty_spaces():
    first_img = cv.imread(f"data/video{VIDEO_NUM}/img{IMG_NUM}.jpg", cv.IMREAD_GRAYSCALE).astype('float')
    cum_array = first_img
    for img_num in range(1+IMAGE_INTERVAL, N_IMAGES+1, IMAGE_INTERVAL):
        print(f"Doing stuff with image {img_num}")
        dir = f"data/video{VIDEO_NUM}/img{img_num}.jpg"
        frame = cv.imread(dir, cv.IMREAD_GRAYSCALE).astype('float')
        cum_array += frame

    cum_array /= (N_IMAGES/IMAGE_INTERVAL)
    cum_array = cum_array.astype(np.uint8)

    # Mask out area of interest defined by CORNERS
    img_mask = np.zeros(cum_array.shape)
    img_mask = cv.fillPoly(img_mask, [CORNERS], 1)
    cum_array = np.where(img_mask == 1, cum_array, 255)


    #Reshape and show on screen!
    cum_array = cv.resize(cum_array, [960, 540])
    cv.imshow("Cumulative distribution over 100 frames", cum_array)

    # Press any button to move on. Pressing 'q' exits the entire script
    key = cv.waitKey()
    if key & 0xFF == ord('q'):
        exit()


def read_video(filename=f"data/video{VIDEO_NUM}/video{VIDEO_NUM}.mp4"):
    cap = cv.VideoCapture(filename)
    assert cap.isOpened(), "Error streaming video"

    counter = 1
    n_images = N_IMAGES
    while cap.isOpened():
        print(f"Reading image {counter}")
        retval, frame = cap.read()
        if retval:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Bruker 3 ganger mindre plass
            cv.imwrite(f'data/video{VIDEO_NUM}/img{counter}.jpg', frame)
            if counter >= n_images:
                break
        counter += 1
    cap.release()


if __name__ == '__main__':
    main()
