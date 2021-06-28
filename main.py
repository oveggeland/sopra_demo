import cv2 as cv
import numpy as np

from visualizer import draw_circles
from annotation import create_annotation_images


def main():
    create_annotation_images()        # Read video and save individual images


if __name__ == '__main__':
    main()
