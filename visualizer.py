import cv2 as cv
import numpy as np

from image_utils import dlt


def draw_circles(img, centers, radius=5, linesize=3):
    #img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for i in range(centers.shape[0]):
        center = (centers[i, 0], centers[i, 1])
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


def create_heat_map(img, corners, width=1000, height=400):
    img[400:600, 1000:1200] = np.ones((200, 200))*255
    img[600:900, 1400:1700] = np.ones((300, 300))*127
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

    cv.imshow("old image", cv.resize(img, (width, height)))
    cv.imshow("new image", new_img)

    #TODO: This is gives ugly black circles where stuff is not transformed, should we do something?

    cv.waitKey()
