import cv2
import numpy as np

from yolo.detection import detect
from visualizer import draw_circles
from empty_spaces import find_empty_spaces
from sandbed import chickens_over_time

if __name__ == '__main__':
    chickens_over_time()
