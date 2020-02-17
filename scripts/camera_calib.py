#******** imports ********#

import numpy as np
import cv2 as cv
from os import path
from glob import glob
import csv
import pickle

#******** global variables ********#

root_dir = path.abspath(
    path.join(path.dirname(path.abspath(__file__)), path.pardir))
imgdir = path.abspath(path.join(root_dir, "images"))
outdir = path.abspath(path.join(root_dir, "output"))
images = glob(path.join(imgdir, "*.jpg"))

#******** chess board variables ********#

chess_rows = 9
chess_cols = 6
chess_sq_width = 25  # mm


# openCv setup

# terminaltion criteria

term_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chess_rows*chess_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:chess_rows, 0:chess_cols].T.reshape(-1, 2)
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


#******** loading files one by one ********#

remaining_images = len(images)  # count of images
img_count = 1
img_shape = None
for frame in images:

    print("remaining images", remaining_images)
    print("reading", frame)
    img = cv.imread(frame, 0)
    img_shape = img.shape[::-1]
    # Find the chess board corners
    print("checking for corners")

    ret, corners = cv.findChessboardCorners(
        img, (chess_rows, chess_cols), None)

    print("done checking for corners")

    # If found, add object points, image points to the list
    if ret == True:
        objpoints.append(objp)
        print("object points added to the list")

        print("checking for sub-pixels")
        corners2 = cv.cornerSubPix(
            img, corners, (11, 11), (-1, -1), term_criteria)
        print("done with sub-pixels")

        imgpoints.append(corners2)
        print("image points added to the list")

#******** saving the files to an output directory ********#

        # img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        # img = cv.drawChessboardCorners(
        #     img, (chess_rows, chess_cols), corners2, ret)
        # imgpath = path.abspath(
        #     path.join(outdir, "img{}.jpg".format(img_count)))
        # print("saving file to", imgpath)
        # cv.imwrite(imgpath, img,)
        # img_count = img_count+1

    else:
        print("unable to find the corners")
    remaining_images = remaining_images-1

#******** calibrating the Camera ********#

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None)

# calculating for error

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

final_error = mean_error/len(objpoints)
print("total error: {}".format(final_error))

# writing to an output file using pickle
output_data_file = path.abspath(path.join(root_dir, "calib"))
if final_error <= 1:  # if the mean error is less than 1 calibration is good enough
    with open(output_data_file, "wb") as file:
        pickle.dump([ret, mtx, dist, rvecs, tvecs], file)
    if path.isfile(output_data_file):
        print("vairables has been saved to external file at", output_data_file)
