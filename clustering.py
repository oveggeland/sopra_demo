import cv2 as cv
import numpy as np
from sklearn import cluster
from visualizer import draw_circles, draw_db_clusters


def DBSCAN_cluster(img):
    # Create clusters of an black/white image
    white_pixels = np.where(img)  # (x-coordinates, y-coordinates)
    coordinates = np.float32(np.column_stack((white_pixels[0], white_pixels[1])))

    db = cluster.DBSCAN(eps=5, min_samples=50).fit(coordinates)
    n_clusters = np.max(db.labels_)+1
    print(f"Number of clusters is {n_clusters}")

    img = draw_db_clusters(img, coordinates, db)

    return img, n_clusters


def affinity_cluster(img):
    white_pixels = np.where(img)  # (x-coordinates, y-coordinates)
    coordinates = np.float32(np.column_stack((white_pixels[0], white_pixels[1])))

    af = cluster.AffinityPropagation(damping=0.99, verbose=True)
    print(1)
    af.fit(coordinates[:, :])
    print(2)
    return img, af.cluster_centers_


def kmeans_cluster(img, k=10):
    white_pixels = np.where(img)    # (x-coordinates, y-coordinates)
    coordinates = np.float32(np.column_stack((white_pixels[0], white_pixels[1])))

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)

    _, _, centers = cv.kmeans(coordinates, k, None, criteria, 1, cv.KMEANS_PP_CENTERS)
    img = draw_circles(img, centers, radius=50)

    return img, centers.shape[0]
