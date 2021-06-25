import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sandbed import find_chickens

# Specify what image/frame to analyze
VIDEO_NUM = 1
IMG_NUM = 1
N_IMAGES = 8
IMAGE_INTERVAL = 1000
FRAME_TIME = 1/19.72

def main():
    #read_video()                            # Read video and save individual images

    chicken_counter = []
    time = []
    for i in range(N_IMAGES):
        img_num = 1+IMAGE_INTERVAL*i
        n_chickens = find_chickens(VIDEO_NUM, img_num)

        chicken_counter.append(n_chickens)
        time.append(img_num*FRAME_TIME)

    plt.plot(time, chicken_counter)
    plt.show()


def read_video():
    filename = f"data/video{VIDEO_NUM}/video{VIDEO_NUM}.mp4"
    cap = cv.VideoCapture(filename)
    assert cap.isOpened(), "Error streaming video"

    counter = 9900
    while cap.isOpened():
        print(counter)
        retval, frame = cap.read()
        if retval:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Bruker 3 ganger mindre plass
            if counter % IMAGE_INTERVAL == 1:
                cv.imwrite(f'data/video{VIDEO_NUM}/img{counter}.jpg', frame)
        if counter == 1 + IMAGE_INTERVAL*(N_IMAGES-1):
            break
        counter += 1
    cap.release()


if __name__ == '__main__':
    main()
