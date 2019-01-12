import cv2
import numpy 
import argparse
import os
import imutils


parser = argparse.ArgumentParser()

parser.add_argument('-i' , '--input' , type=str , default='images/' ,
						help='Directory to images')

FLAGS , unparsed = parser.parse_known_args()

imagePath = os.path.join(FLAGS.input , 'city.jpg')

image = cv2.imread(imagePath)
image = imutils.resize(image , width=700)
img=image.copy()
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

#create SIFT object
sift = cv2.xfeatures2d.SIFT_create()
keypoints , descriptors = sift.detectAndCompute(gray , None)

img = cv2.drawKeypoints(image = image , outImage=img , keypoints = keypoints,
						flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS , color=(51,163,236))

cv2.imshow('Original' , image)
cv2.imshow("SIFT" , img)

while True:

	if cv2.waitKey(1)==32:
		break

cv2.destroyAllWindows()