# import the necessary packages
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
 
camera = cv2.VideoCapture(0)

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

while True:

	(grabbed, frame) = camera.read()
	image = frame

	sz = 500
	ratio = (sz + 0.0) / image.shape[1]
	dim = (sz, int(image.shape[0] * ratio))
	image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	laplacian = cv2.Laplacian(gray,cv2.CV_8U)
	sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=5)
	sobely = cv2.Sobel(gray,cv2.CV_8U,0,1,ksize=5)
	sharrx = cv2.Scharr(gray, cv2.CV_8U, 1, 0)
	sharry = cv2.Scharr(gray, cv2.CV_8U, 0, 1)

	cv2.imshow("Image", display([sharrx, sharry, image, sobelx, sobely, gray]))

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

