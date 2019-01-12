import cv2
import numpy as np 
import os 
import argparse
import imutils



parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input' , type=str , default= 'images/' ,
							 help='The directory where input images are stored')


FLAGS, unparsed = parser.parse_known_args()
print("[INFO]Reading image...")
inputpath = os.path.join(FLAGS.input , 'chess-board.jpg')

image = cv2.imread(inputpath)
image = imutils.resize(image , width=400)
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray , 2 , 23 , 0.04)
img = image.copy()
img[dst>0.01 * dst.max()] = [0,0,255]
#print("[INFO]Loading processed image...")
while True:

	cv2.imshow("corner" , image)
	cv2.imshow("original" , img)

	if cv2.waitKey(1)==32:
		break 


cv2.destroyAllWindows()
