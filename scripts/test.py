import cv2 as cv
from os import path
# import pickle

root_dir = path.abspath(
    path.join(path.dirname(path.abspath(__file__)), path.pardir))

# datafile = path.abspath(path.join(root_dir, "calib.pickle"))
# datafile1 = path.abspath(path.join(root_dir, "calib"))
# ret = None
# mtx = None
# dist = None
# rvecs = None
# tvecs = None

# ret1 = None
# mtx1 = None
# dist1 = None
# rvecs1 = None
# tvecs1 = None

# with open(datafile, "rb") as file:
#     ret, mtx, dist, rvecs, tvecs = pickle.load(file)

# with open(datafile1, "rb") as file:
#     ret1, mtx1, dist1, rvecs1, tvecs1 = pickle.load(file)

# print(mtx)
# print(mtx1)

print(cv.useOptimized())
img = cv.imread(path.join(root_dir, "test_img.jpg"))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 100, 120)

cv.namedWindow("img", cv.WINDOW_NORMAL)
cv.resizeWindow("img", 1024, 700)
cv.imshow("img", edges)
cv.waitKey(0)
cv.destroyAllWindows()
