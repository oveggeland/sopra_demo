import cv2 as cv
import numpy as np

"""
Create a masked image, where all pixels outside a defined area is set to zero.

Params:
    img - Image to mask out
    pts - Points to define a polygon. Everything outside this polygon is set to zero. 

Returns:
    corners - Masked image. 
"""
def mask_image(img, pts):
    img_mask = np.zeros(img.shape)
    cv.fillPoly(img_mask, [pts], 1)
    return img * img_mask.astype(np.uint8)


"""
Crop image just big enough to include all points given by pts.

Params:
    img - Image to crop
    pts - Points defining the outer values of the image crop.

Returns:
    cropped
"""
def crop_image(img, pts):
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    cropped_image = img[min_y:max_y, min_x:max_x]
    return cropped_image


"""
Performs a direct linear transformation.
Creates a transformation between to 3d planes based on at least four homogenous correspondences between the planes.

Params:
    xy - Homogenous coordinates of at least four co-planar points in the first plane
    XY - Homogenous coordinates of at least four co-planar points in the second plane

Returns:
    H - A transformation matrix from XY to xy (xy = H @ XY)
"""
def dlt(xy, XY):
    n_points = xy.shape[1]
    assert n_points == 4, "DLT should get four point correspondence to work"
    A = np.zeros((8, 9))
    for i in range(n_points):
        A[2*i:2*(i+1), :] = np.array([
            [XY[0, i], XY[1, i], 1, 0, 0, 0, -XY[0, i]*xy[0, i], -XY[1, i]*xy[0, i], -xy[0, i]],
            [0, 0, 0, XY[0, i], XY[1, i], 1, -XY[0, i]*xy[1, i], -XY[1, i]*xy[1, i], -xy[1, i]]
        ])

    _, _, V = np.linalg.svd(A)
    V = V.T
    h = V[:, -1]  # Last column of V
    H = np.reshape(h, [3, 3])

    return H

