import cv2 as cv

# Specify what image/frame to analyze
VIDEO_NUM = 1
IMG_NUM = 1
N_IMAGES = 50
IMAGE_INTERVAL = 200


def create_annotation_images(target_dir='data/annotation/images'):
    cap = cv.VideoCapture(f'data/video{VIDEO_NUM}/video{VIDEO_NUM}.mp4')
    assert cap.isOpened(), "Error streaming video"

    counter = IMG_NUM
    while cap.isOpened():
        print(counter)
        retval, og = cap.read()
        if retval:
            if (counter - IMG_NUM) % IMAGE_INTERVAL == 0:
                frame = og[360:940, 890:1630]
                cv.imwrite(f'{target_dir}/{counter}.jpg', frame)
                print("hei")
            if counter >= N_IMAGES*IMAGE_INTERVAL+IMG_NUM:
                break
        counter += 1
    cap.release()
