import cv2
import numpy as np

from yolo.detection import detect
from visualizer import draw_circles

if __name__ == '__main__':
    print("Hello world")
    detect(line_thickness=1, hide_conf=True, hide_labels=True)

    img = np.zeros((500, 500))
    centers = np.array([
        [250, 250]
    ])
    draw_circles(img, centers)
    cv2.imshow("test", img)
    cv2.waitKey()
