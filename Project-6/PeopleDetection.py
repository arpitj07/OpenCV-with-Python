import cv2
import numpy as np 
import imutils
import argparse
import os
from imutils.object_detection import non_max_suppression
from imutils import paths

parser = argparse.ArgumentParser()

parser.add_argument("-i" , "--input" , default="images/" , type=str , required=True,help='path to images')
#parser.add_argument("-I" , "--image" , type=str , help='input image for detection')
FLAGS, args = parser.parse_known_args()


def draw_person(image , person):
	x,y,w,h = person
	cv2.rectangle(image , (x,y) , (x+w , y+h) , (0,0,255) ,2)


def detect_person(image):
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	rects , weights = hog.detectMultiScale(image , winStride=(4,4) , padding=(8,8), scale=1.05)
	return rects, weights



#DETECTING PEOPLE IN IMAGES
counter=0
for file in paths.list_images(FLAGS.input):
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy

	image = cv2.imread(str(file))
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()

	# detect people in the image
	(rects, weights) = detect_person(image)

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
	filename = file[file.rfind("/") + 1:]
	print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))
	counter+=1
	# show the output images
	cv2.imshow("Before NMS", orig)
	cv2.imshow("After NMS", image)
	cv2.imwrite("Image_" + str(counter) + ".jpg" , image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	