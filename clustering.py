import cv2 as cv
import numpy as np
from sklearn import cluster
from visualizer import draw_circles, draw_db_clusters


def DBSCAN_cluster(img):
    # Create clusters of an black/white image
    white_pixels = np.where(img)  # (x-coordinates, y-coordinates)
    coordinates = np.float32(np.column_stack((white_pixels[0], white_pixels[1])))

    db = cluster.DBSCAN(eps=3, min_samples=15).fit(coordinates)
    unique, counts = np.unique(db.labels_, return_counts=True)
    valid_clusters = np.where(counts[1:] > 500)[0]
    n_chickens = valid_clusters.size

    print(f"Number of clusters is {n_chickens}")

    img = draw_db_clusters(img, coordinates, db, valid_clusters)
    key = cv.waitKey()
    if key & 0xFF == ord('q'):
        exit()

    return img, n_chickens


def kmeans_cluster(img, k=10):
    white_pixels = np.where(img)    # (x-coordinates, y-coordinates)
    coordinates = np.float32(np.column_stack((white_pixels[0], white_pixels[1])))

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)

    _, _, centers = cv.kmeans(coordinates, k, None, criteria, 1, cv.KMEANS_PP_CENTERS)
    img = draw_circles(img, centers, radius=50)

    return img, centers.shape[0]
