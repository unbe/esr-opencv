# import the necessary packages
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
 
camera = cv2.VideoCapture(0)

while True:

	(grabbed, frame) = camera.read()
	image = frame

	sz = 500
	ratio = (sz + 0.0) / image.shape[1]
	dim = (sz, int(image.shape[0] * ratio))
	image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)	

	planes = cv2.split(converted)
	for i in range(len(planes)):
		planes[i] = cv2.cvtColor(planes[i], cv2.COLOR_GRAY2RGB)

	vis1 = np.concatenate((image, planes[0]), axis=1)
	vis2 = np.concatenate((planes[1], planes[2]), axis=1)
	vis = np.concatenate((vis1, vis2), axis=0)
	cv2.imshow("Image", vis)

"""

plt.subplot(2,2,1),plt.imshow(image,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
"""

cv2.waitKey(0)
