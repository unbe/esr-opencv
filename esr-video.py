# import the necessary packages
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
 

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--camera", action='store_true', help = "Use camera")
ap.add_argument("-i", "--image", help = "Use image")
args = vars(ap.parse_args())

def display(imgs):
	nx = int(np.ceil(np.sqrt(len(imgs))))
	ny = np.ceil((len(imgs) + 0.0)/nx)
	rows = []
	for y in range(int(ny)):
		irow = imgs[y * nx : (y + 1) * nx]
		while len(irow) < nx:
			irow.append(irow[-1])
		for i in range(len(irow)):
			if len(irow[i].shape) != 3 or irow[i].shape[2] != 3:
				irow[i] = cv2.cvtColor(irow[i], cv2.COLOR_GRAY2BGR)
		row = np.concatenate(irow, axis=1)
		rows.append(row)
	return np.concatenate(rows, axis=0)

def process(image):
	sz = 500
	ratio = (sz + 0.0) / image.shape[1]
	dim = (sz, int(image.shape[0] * ratio))
	image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	bila = cv2.bilateralFilter(image, 11, 17, 17)
	gray = cv2.cvtColor(bila, cv2.COLOR_BGR2GRAY)

	#laplacian = cv2.Laplacian(gray,cv2.CV_8U)
	#sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=5)
	#sobely = cv2.Sobel(gray,cv2.CV_8U,0,1,ksize=5)
	# sharrx = cv2.Scharr(gray, cv2.CV_8U, 1, 0)
	# blur = cv2.GaussianBlur(gray, (3, 3), 0)
	#sharry = cv2.Scharr(gray, cv2.CV_8U, 0, 1)
	canny = cv2.Canny(gray, 25, 50)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
	#canny = cv2.dilate(canny, kernel, iterations = 1)
	#canny = cv2.erode(canny, kernel, iterations = 1)
	
	(cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	goodCnts = []
	badCnts = []
	# loop over our contours
	for c in cnts:
		if cv2.contourArea(c) < 1000:
			continue
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)
#		print len(approx)
 
		# if our approximated contour has four points, then
		# we can assume that we have found our screen
		if len(approx) == 4:
			goodCnts.append(approx)
		else:
			approx = cv2.approxPolyDP(c, 0.01 * peri, False)
			badCnts.append(approx)

	#cv2.drawContours(image, cnts, -1, (0, 0, 255), 1)
	cv2.drawContours(image, goodCnts, -1, (0, 255, 0), 2)
	cv2.drawContours(image, badCnts, -1, (128, 128, 0), 1)
		

#	lines = cv2.HoughLinesP(canny, 1, np.pi/180, 50, 50, 10)
#	if lines is not None:
#    		for x1,y1,x2,y2 in lines[0]:        
#        		cv2.line(image,(x1,y1),(x2,y2),(0,255,0),1)


	print len(goodCnts)
	cv2.imshow("Image", display([image, bila, gray, canny]))


if not args.get("camera", False):
	image = cv2.imread(args["image"])
	process(image)
	cv2.waitKey(0)
	exit(0)
else:
	camera = cv2.VideoCapture(0)

while True:
	(grabbed, frame) = camera.read()
	process(frame)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

