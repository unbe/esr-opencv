# import the necessary packages
import numpy as np
import argparse
import cv2

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image file")
args = vars(ap.parse_args())


# load the image and convert it to grayscale
image = cv2.imread(args["image"])

orig = image.copy()

ratio = 400.0 / image.shape[1]
dim = (400, int(image.shape[0] * ratio))
 
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

image = cv2.blur(image, (3, 3))

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower = np.array([12, 5, 0], dtype = "uint8")
upper = np.array([35, 255, 255], dtype = "uint8")
mask = cv2.inRange(hsv, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#mask = cv2.erode(mask, kernel, iterations = 1)
#mask = cv2.dilate(mask, kernel, iterations = 1)

#mask = cv2.Canny(mask, 30, 200)

#cv2.imshow('Image - %d %d %d' % (d, sc, ss), mask)
#cv2.imwrite('mask.%d.%d.%d.%d.%d.png' % (d, ss, sc, mn, mx), mask)
cv2.imshow("images", mask)

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
#mask = cv2.erode(mask, kernel, iterations = 2)
#mask = cv2.dilate(mask, kernel, iterations = 2)

#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.bilateralFilter(gray, 11, 17, 17)
#cv2.imshow('Image', gray)

#edged = cv2.Canny(gray, 30, 200)
#cv2.imshow('Image - edged', edged)


cv2.waitKey(0)
