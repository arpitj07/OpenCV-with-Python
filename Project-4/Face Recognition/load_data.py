import cv2
import numpy as np 
import os
import sys

def load_images(path , sz=None):

	c=0
	x,y=[],[]
	for root , directory , file in os.walk(path):
		for subdirname in directory:
			subject_path = os.path.join(root , subdirname)

			for filename in os.listdir(subject_path):

				try:

					if (filename == ".directory"):
						continue
					filepath = 	os.path.join(subject_path,filename)
					im = cv2.imread(os.path.join(subject_path,filename) , cv2.IMREAD_GRAYSCALE)

					if (im is None):
						print("image " + filepath + " is none")
					# resize to given size (if given)
					if (sz is not None):
						im = cv2.resize(im , (200,200))
					image = np.array(im)
					x.append(im)
					(y.append(c))

				except IOError as err:
					print("I/O error({}) : {}".format(errno , strerror))

				except:
					print("Unexpected error" , sys.exc_info()[0])
					raise
			
			c = c+1
	
	return [x,y]	

