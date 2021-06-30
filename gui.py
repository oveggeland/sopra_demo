import cv2 as cv
import numpy as np


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(param[0], (x, y), 20, (255, 0, 255), 5)
        mouseX, mouseY = x, y
        print(f'You clicked on ({x}, {y})')
        param[1].append(np.array([x, y]))


def request_corners(img):
    cv.namedWindow('image')
    corners = []
    cv.setMouseCallback('image', draw_circle, (img, corners))

    print("Double click on corners. To finish, please press q button :) ")
    while True:
        cv.imshow('image', img)
        k = cv.waitKey(20) & 0xFF
        if k == ord('q'):
            break

    return np.array(corners)