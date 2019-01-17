import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import argparse
import os 


parser = argparse.ArgumentParser()
parser.add_argument("-i" , "--input" , type=str , help = 'directory to input images')
FLAGS , unparsed = parser.parse_known_args()

image1= cv2.imread(FLAGS.input + '/adidas.jpg' , cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture(0)

while True:

	ret , frame = cap.read()
	gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

	orb = cv2.ORB_create()

	kp1 , ds1 = orb.detectAndCompute(image1 , None)
	kp2 , ds2 = orb.detectAndCompute(gray , None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck=False)
	matches = bf.knnMatch(ds1 , ds2 , k=2)

	image = cv2.drawMatchesKnn(image1 , kp1 , gray , kp2 , matches , gray , flags=2)

	cv2.imshow("KNN",image) #, plt.show()

	if cv2.waitKey(1)==32:
		break

cap.release()
cv2.destroyAllWindows()

