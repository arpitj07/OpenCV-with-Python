import cv2
import numpy as np 
import sys 
import os
import imutils
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-i" , "--input" , type=str , help = 'directory to input images')
FLAGS , unparsed = parser.parse_known_args()

image2= cv2.imread(FLAGS.input + '/Breaking Bad.jpg' , cv2.IMREAD_GRAYSCALE)
image1= cv2.imread(FLAGS.input + '/Breaking Bad-2.jpg' , cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
kp1 , ds1 = orb.detectAndCompute(image1 , None)
kp2 , ds2 = orb.detectAndCompute(image2 , None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck=True)
matches = bf.match(ds1 , ds2)
matches = sorted(matches , key = lambda x: x.distance)

image = cv2.drawMatches(image1 , kp1 , image2 , kp2 , matches[:40] , image2 , flags=2)

plt.imshow(image), plt.show()
cv2.imwrite("ORB.jpg" , image)

