import cv2 as cv
import numpy as np


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(param[0], (x, y), param[2], param[3], param[4])
        mouseX, mouseY = x, y
        print(f'You clicked on ({x}, {y})')
        param[1].append(np.array([x, y]))


def request_corners(img, radius=10, color=(255, 0, 0), line_thickness=2, num=None):
    cv.namedWindow('image')
    corners = []
    cv.setMouseCallback('image', draw_circle, [img, corners, radius, color, line_thickness])

    print(f"Click on corners. z button regrets the last corner. q to finish.")
    while True:
        cv.imshow('image', img)
        k = cv.waitKey(20) & 0xFF
        if k == ord('q') or len(corners) == num:
            break
        if k == ord('z'):
            del corners[-1]
            print(corners)
            cv.circle(img, (mouseX, mouseY), 20, (0, 0, 255), 5)

    return np.array(corners)
