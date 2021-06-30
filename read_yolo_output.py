import numpy as np
import pandas as pd
import cv2 as cv
from image_utils import mask_image

def return_center(path = "yolo/runs/detect/exp/labels/1001_jpg.rf.5746260f252219fdb0f254e167edacc2.txt", as_pixel = True):
    """Return center of yolo detected objects as a numpy array with x, y coordinates"""
    data = pd.read_csv(path, sep=" ", header=None)
    data.columns = ["class", "x", "y", "w", "h."]
    center_array = data.to_numpy()[:, 1:3]
    if as_pixel:
        center_array = center_array * [w, h]
    return(center_array)

def convert_norm_to_pixel(center_array, w, h):
    center_array * [w, h]
#coutn in box

corners = np.array([
    [0, 0],
    [0, 300],
    [300, 300],
    [300, 0]
])

img = cv.imread("yolo/runs/detect/exp/1001_jpg.rf.5746260f252219fdb0f254e167edacc2.jpg")
#cv.imshow("img", mat = img)
h, w, d = img.shape
print(w)

center_detect = return_center("yolo/runs/detect/exp/labels/1001_jpg.rf.5746260f252219fdb0f254e167edacc2.txt")
print(center_detect*[w,h])

#centers = np.zeros(img.shape)




mask = mask_image(img, corners)
#cv.imshow("mask", mask)

key = cv.waitKey()
if key & 0xFF == ord('q'):
    exit()