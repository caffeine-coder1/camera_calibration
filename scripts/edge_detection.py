import cv2 as cv

# opening the images

img_original = cv.imread(
    "/home/karthik/work/CTO_team/camera_calibration/scripts/chessboard.jpg")
img = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (7, 7), 0)
img = cv.Canny(img, 50, 100)
img = cv.dilate(img, None, iterations=2)
# img = cv.erode(img, None, iterations=1)
cv.namedWindow("original", cv.WINDOW_NORMAL)
cv.namedWindow("edged", cv.WINDOW_NORMAL)
cv.imshow("original", img_original)
cv.imshow("edged", img)
cv.waitKey(0)
cv.destroyAllWindows()
