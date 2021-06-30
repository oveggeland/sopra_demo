import cv2 as cv
import numpy as np
from scipy.optimize import fsolve
from read_yolo_output import return_center
from gui import request_corners


# Residual function to feed nonlinear solver in search of a good line parameterization for points x0 and x1
def line_residual(x0, x1, theta, rho):
    x_tilde = np.array([
        [x0[0], x0[1], 1],
        [x1[0], x1[1], 1]
    ])
    l = np.array([np.cos(theta), np.sin(theta), -rho])
    return x_tilde @ l


# Non linear search for line parameters corresponding to points x0 and x1
def get_line_params(x0, x1):
    res_func = lambda p: line_residual(x0, x1, p[0], p[1])
    while True:
        p0 = np.random.rand(2)
        p, info, _, _ = fsolve(res_func, p0, full_output=True)
        if info['fvec'].all() < 0.001:
            break
    return np.array([p[0], p[1], x0, x1])


# Return all points that are sufficiently close to all lines
def get_points_near_lines(points, lines, threshold=15):
    indices = []
    for i in range(points.shape[0]):
        for line in lines:
            l = np.array([np.cos(line[0]), np.sin(line[0]), -line[1]])
            x_tilde = np.array([points[i, 0], points[i, 1], 1])
            if abs(x_tilde @ l) < threshold:
                d1 = np.linalg.norm(points[i, :] - line[2])
                d2 = np.linalg.norm(points[i, :] - line[3])
                if d1 < np.linalg.norm(line[2]-line[3])+threshold\
                        and d2 < np.linalg.norm(line[2]-line[3])+threshold:
                    indices.append(i)
                break
    return points[indices, :]


if __name__ == "__main__":
    line = get_line_params(np.array([1, 4]), np.array([1, -1]))
    if line[0] < 1e-10 and abs(line[1]-1) < 1e-10:
        print("Line function working!")

    img = cv.imread("yolo/runs/detect/exp/1001_jpg.rf.5746260f252219fdb0f254e167edacc2.jpg")
    points = request_corners(img)

    for i in range(4):
        cv.line(img, points[i-1], points[i], (255, 0, 0), 3)
    cv.imshow("lines", img)

    #Load centers
    centers = return_center()
    debug = 1

    lines = []
    for i in range(4):
        lines.append(get_line_params(points[i-1], points[i]))
    sitters = get_points_near_lines(centers, lines).astype(int)

    for chicken in sitters:
        cv.circle(img, (chicken[0], chicken[1]), 2, (0, 255, 255), 2)

    cv.imshow("Sitters", cv.resize(img, (720, 720)))
    cv.waitKey()
