import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sandbed import find_chickens

# Specify what image/frame to analyze
VIDEO_NUM = 1
IMG_NUM = 1
N_IMAGES = 5

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


def main():
    read_video()                            # Read video and save individual images
    find_chickens(VIDEO_NUM, IMG_NUM)

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
