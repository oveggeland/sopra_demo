import cv2 as cv
import numpy as np

from image_utils import dlt


"""
Draw circles on an image and return a copy

Params:
    img - Image to draw circles on
    centers - Numpy array of all circle centers
    color - Circle color
    radius - Circle radius
    line_thickness - Circle line thickness
"""
def draw_circles(img, centers, color=(127, 127, 127), radius=5, line_thickness=3):
    for i in range(centers.shape[0]):
        center = (centers[i, 0], centers[i, 1])
        cv.circle(img, center, radius, color, line_thickness)


"""
Draw clusters obtained from a DBSCAN. 

Params:
    img - Image to draw clusters on
    coordinates - List of coordinates used when clustering
    db - DBSCANNER object containing cluster information
    valid_clusters - List of cluster ID's that are considered valid clusters (due to size)
"""
def draw_db_clusters(img, coordinates, db, valid_clusters):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for c in valid_clusters:   # Iterate over each cluster
        color = (list(np.random.choice(range(256), size=3)))
        cluster_indices = np.where(db.labels_ == c)[0]
        cluster_coordinates = coordinates[cluster_indices].astype('int')
        for i in range(cluster_indices.size):
            y = int(cluster_coordinates[i, 0])
            x = int(cluster_coordinates[i, 1])
            img[y, x] = color

    cv.imshow("Colored clusters", img)
    return img


"""
Transforms an image area into a rectangular heat map using a direct linear transform.

Params:
    img - Image source 
    corners - Corners defining an area to transform
    width - Width of heat map
    height - Height of heat map

Returns:
    new_img - Image of the transformed heat map
"""
def create_heat_map(img, corners, width=1000, height=400):
    XY = np.array([
        [width, width, 0, 0],
        [0, height, height, 0]
    ])
    xy = corners.T
    H = dlt(xy, XY)

    new_img = np.ones((height, width))
    for x in range(np.min(xy[0, :]), np.max(xy[0, :])):
        for y in range(np.min(xy[1, :]), np.max(xy[1, :])):
            new_coordinate = np.linalg.solve(H, np.array([x, y, 1]))
            new_x = int(new_coordinate[0] // new_coordinate[2])
            new_y = int(new_coordinate[1] // new_coordinate[2])
            if new_x < 0 or new_y < 0 or new_y >= height or new_x >= width:
                pass
            else:
                new_img[new_y, new_x] = img[y, x]

    return new_img
