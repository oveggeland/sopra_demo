import numpy as np
import pandas as pd
import cv2 as cv

"""
Returns the center of all boxes from a yolo prediction. 

Params:
    file_name - Name of the prediction file to use
    as_pixel - Specifies whether box-center coordinates should be returned as pixel values or normalized values

Returns:
    center_array - Numpy array of all box-centers
"""
def return_centers(file_name = "1001_jpg.rf.5746260f252219fdb0f254e167edacc2", as_pixel = True):
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
    return center_array


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
