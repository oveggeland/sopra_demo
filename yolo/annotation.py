import cv2 as cv
import numpy as np
from gui import request_corners
from image_utils import crop_image

# Specify what image/frame to analyze
VIDEO_NUM = 1
IMG_NUM = 1
N_IMAGES = 1
IMAGE_INTERVAL = 200


def create_annotation_images(target_dir='../data/annotation/images'):
    cap = cv.VideoCapture(f'../data/video{VIDEO_NUM}/video{VIDEO_NUM}.mp4')
    assert cap.isOpened(), "Error streaming video"

    base_name = input("Hva vil du kalle annoteringsbildene?")

    counter = IMG_NUM
    while cap.isOpened():
        print(counter)
        retval, frame = cap.read()

        if counter == IMG_NUM:
            width, height = 720, 540
            corners = request_corners(cv.resize(frame.copy(), (width, height)))
            corners = (corners.astype(float)*[frame.shape[1]/width, frame.shape[0]/height]).astype(int)
            cv.destroyAllWindows()

        if retval:
            if (counter - IMG_NUM) % IMAGE_INTERVAL == 0:

                frame = crop_image(frame, corners)
                cv.imwrite(f'{target_dir}/{base_name}{counter}.jpg', frame)
                print("hei")
            if counter >= N_IMAGES*IMAGE_INTERVAL+IMG_NUM:
                break
        counter += 1
    cap.release()


if __name__ == "__main__":
    create_annotation_images()
