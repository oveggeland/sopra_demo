import cv2 as cv
import numpy as np

"""
Mouse event function. Draws a circle on mouseclick.

Params:
    event - Type of mouse event
    x, y - Coordinates of mouse
    flags - Unused
    param[0] - Image where mouse event took place
    param[1] - Table to store mouse coordinates on event
    param[2] - Radius of circle
    param[3] - Color of circle
    param[4] - Linethickness of circle
"""
def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(param[0], (x, y), param[2], param[3], param[4])
        mouseX, mouseY = x, y
        print(f'You clicked on ({x}, {y})')
        param[1].append(np.array([x, y]))


"""
GUI to request a set of points by having the user click on an image. 

Params:
    img - Image to be used for choosing coordinates
    radius - Radius of circle to confirm a click on image
    color - Color of circle
    line_thickness - Thickness of circle
    num - Number of points to request

Returns:
    corners - Numpy array containing the coordinates clicked on by user
"""
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
