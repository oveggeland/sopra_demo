import cv2 as cv

VIDEO_NUM = 1
IMG_NUM = 1
N_IMAGES = 8
IMAGE_INTERVAL = 1000


"""
Reads a video and saves .jpg files from specified frames

Params:
    video_num - ID to define what video to read
    img_num - First frame to save
    n_images - Number of frames to save
    image_interval - Number of frames to skip between saves
"""
def read_video(video_num=VIDEO_NUM, img_num=IMG_NUM, n_images=N_IMAGES, image_interval=IMAGE_INTERVAL):
    filename = f"data/video{video_num}/video{video_num}.mp4"
    cap = cv.VideoCapture(filename)
    assert cap.isOpened(), "Error streaming video"

    counter = img_num
    while cap.isOpened():
        print(counter)
        retval, frame = cap.read()
        if retval:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Bruker 3 ganger mindre plass
            if counter % image_interval == 1:
                cv.imwrite(f'data/video{video_num}/img{counter}.jpg', frame)
        if counter == 1 + image_interval*(n_images-1):
            break
        counter += 1
    cap.release()
