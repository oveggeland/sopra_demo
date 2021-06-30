import numpy as np
import pandas as pd
import cv2 as cv
from image_utils import mask_image

from visualizer import draw_circles

def return_center(file_name = "1001_jpg.rf.5746260f252219fdb0f254e167edacc2", as_pixel = True):
    """Return center of yolo detected objects as a numpy array with x, y coordinates either normalized or as pixels"""

    path_txt = "yolo/runs/detect/exp/labels/" + file_name + ".txt"
    path_img = "yolo/runs/detect/exp/" + file_name + ".jpg"
    img = cv.imread(path_img)
    print(path_img)
    h, w, d = img.shape
    print(w)
    data = pd.read_csv(path_txt, sep=" ", header=None)
    data.columns = ["class", "x", "y", "w", "h."]
    center_array = data.to_numpy()[:, 1:3]
    if as_pixel:
        center_array = center_array * [w, h]
        center_array = center_array.astype(int)
    return(center_array)

def chickens_in_area(corners, file_name = "1001_jpg.rf.5746260f252219fdb0f254e167edacc2"):
    path_img = "yolo/runs/detect/exp/" + file_name + ".jpg"

    img = cv.imread(path_img)
    h, w, d = img.shape
    print(w)
    center_detect = return_center(file_name, as_pixel=True)
    centers_matrix = np.zeros(img.shape[0:2])

    # label all coordinate positions with one
    centers_matrix[center_detect[:, 0], center_detect[:, 1]] = 1
    # Mask area of interest
    mask= mask_image(centers_matrix, corners)

    #convert positions back to coordinates
    coordinates_of_interest = np.where(mask==1)
    coordinates_of_interest = np.transpose(np.vstack((coordinates_of_interest[0], coordinates_of_interest[1])))

    # Draw circles at the captured chicken
    chicken_in_area_img = draw_circles(img, coordinates_of_interest, radius=1)
    count = len(coordinates_of_interest[0])
    return(chicken_in_area_img, count)

if __name__ == '__main__':

    corners = np.array([
        [50, 60],
        [350, 60],
        [350, 350],
        [50, 350]
    ])
    chickens_in_area(corners)
    key = cv.waitKey()
    if key & 0xFF == ord('q'):
        exit()
