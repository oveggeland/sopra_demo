import cv2 as cv
import numpy as np
from sklearn import cluster
from visualizer import draw_circles, draw_db_clusters

"""
Finds clusters of white pixels in black/white image.

Params:
    img - black and white input img
    min_size - minimum size of a cluster

Returns:
    img - img with clusters colored in random colors
    n_clusters - number of clusters found
"""
def DBSCAN_cluster(img, min_size=500):
    white_pixels = np.where(img)
    coordinates = np.float32(np.column_stack((white_pixels[0], white_pixels[1])))

    db = cluster.DBSCAN(eps=3, min_samples=15).fit(coordinates)
    unique, counts = np.unique(db.labels_, return_counts=True)
    valid_clusters = np.where(counts[1:] > min_size)[0]
    n_clusters = valid_clusters.size

    img = draw_db_clusters(img, coordinates, db, valid_clusters)

    return img, n_clusters


"""
Finds centers of k clusters of white pixels in black/white image.

Params:
    img - Black and white input image 
    k - Number of clusters to locate

Returns:
    img - img with clusters colored in random colors
    k - Number of clusters to locate (included by convention with DBSCAN)
"""
def kmeans_cluster(img, k=10):
    white_pixels = np.where(img)    # (x-coordinates, y-coordinates)
    coordinates = np.float32(np.column_stack((white_pixels[0], white_pixels[1])))

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)

    _, _, centers = cv.kmeans(coordinates, k, None, criteria, 1, cv.KMEANS_PP_CENTERS)
    img = draw_circles(img, centers, radius=50)

    return img, k
