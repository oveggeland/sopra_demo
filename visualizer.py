import cv2 as cv
import numpy as np


def draw_circles(img, centers, radius=5, linesize=3):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for i in range(centers.shape[0]):
        center = (centers[i, 1], centers[i, 0])
        cv.circle(img, center, radius, (0, 255, 0), linesize)
    cv.imshow('Circles', img)
    return img


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
