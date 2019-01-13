import cv2 
import numpy as np 
import sys 
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("-i" , '--input' , type=str ,
						 help= 'Directory to input images')

parser.add_argument('-al' , '--algorithm' , type =str, help= 'choose either SURF or SIFT' )


FLAGS , unparsed = parser.parse_known_args()
#imgPath = os.path.join(FLAGS.input , "city2.jpg")


image = cv2.imread(FLAGS.input + '/city2.jpg')
img = image.copy()
alg = FLAGS.algorithm

def featuredetect(algorithm):

	if algorithm =="SIFT":
		return cv2.xfeatures2d.SIFT_create()

	if algorithm == 'SURF':
		return cv2.xfeatures2d.SURF_create(4000)

gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
fd_alg = featuredetect(alg)
keypoints , descriptors = fd_alg.detectAndCompute(gray , None)

img = cv2.drawKeypoints(image=img  , outImage= img , keypoints=keypoints , flags=4 , color=(51,163,236))

cv2.imshow("original" , image)
cv2.imshow("detected", img)

if alg == 'SURF':
	cv2.imwrite("detected-SURF.jpg" , img)
if alg == 'SIFT':
	cv2.imwrite("detected-SIFT.jpg" , img)

while True:
	if cv2.waitKey(1)==32:
		break

cv2.destroyAllWindows()