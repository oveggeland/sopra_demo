import cv2 as cv
import numpy as np

from clustering import cluster
from visualizer import draw_circles

VIDEO_NUM = 1


def main():
    read_video()


def read_video(filename=f"data/video{VIDEO_NUM}.mp4"):
    cap = cv.VideoCapture(filename)
    assert cap.isOpened(), "Error streaming video"

    while cap.isOpened():
        retval, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)       # Bruker 3 ganger mindre plass
        frame = cv.resize(frame, (1024, 720))               # Tilpass en normal pc skjerm

        # Smooth images
        blurred_img = cv.bilateralFilter(frame, 20, 20, 10)
        blur_diff = blurred_img-frame
        # Apply thresholding to remove noise
        threshold_img = cv.adaptiveThreshold(blurred_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                                  cv.THRESH_BINARY, 51, -10)

        centers = cluster(threshold_img, K=300, draw=True)
        draw_circles(frame, centers, radius=20)

        if retval:
            cv.imshow('Frame', frame)
            cv.imshow('Blurred', blurred_img)
            cv.imshow('Blur diff', blur_diff)
            cv.imshow('Threshold', threshold_img)


            if cv.waitKey(10000) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
