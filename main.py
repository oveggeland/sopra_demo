import cv2 as cv
import numpy as np

from clustering import cluster
from visualizer import draw_circles

# Specify what image/frame to analyze
VIDEO_NUM = 1
IMG_NUM = 1


def main():
    read_video()        # Read video and save individual images
    analyze_image()     # Analyze image (default is specified by VIDEO_NUM and IMG_NUM)


def analyze_image(dir=f'data/video{VIDEO_NUM}/img{IMG_NUM}.jpg'):
    img = cv.imread(dir)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (1024, 720))  # Tilpass en normal pc skjerm

    # Smooth images
    blurred_img = cv.bilateralFilter(img, 20, 20, 10)

    # Apply white/black thresholding
    threshold_img = cv.adaptiveThreshold(blurred_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, -10)

    # Cluster white areas to detect hens
    centers = cluster(threshold_img, K=300, draw=True)

    # Show different results results
    draw_circles(img, centers, radius=20)       # Detection visualizer
    cv.imshow('Frame', img)
    # cv.imshow('Blurred', blurred_img)
    cv.imshow('Threshold', threshold_img)

    key = cv.waitKey()
    if key & 0xFF == ord('q'):
        exit()


def read_video(filename=f"data/video{VIDEO_NUM}/video{VIDEO_NUM}.mp4"):
    cap = cv.VideoCapture(filename)
    assert cap.isOpened(), "Error streaming video"

    counter = 1
    n_images = 5
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
