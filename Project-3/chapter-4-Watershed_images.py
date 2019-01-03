import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import imutils


frame = cv2.imread("./Images/leaf.jpg")
frame = imutils.resize(frame , width =200)
gray= cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
_ , thresh = cv2.threshold( gray , 0 , 255 , cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


kernel = np.ones((3,3) , np.uint8)

# removing noice from image by dilating and eroding the image
opening = cv2.morphologyEx( thresh , cv2.MORPH_OPEN , kernel , iterations =2)

#dilating the morphology output gives assurity of the background in image
sure_bg = cv2.dilate(opening , kernel , iterations =3)

# applying distace Transform to get the probability of the foreground area in image
dist = cv2.distanceTransform( opening , cv2.DIST_L2 , 5)

_, sure_fg = cv2.threshold(dist , 0.7*dist.max() , 255, 0)

sure_fg = np.uint8(sure_fg)
#determining area between foreground and background
unknown = cv2.subtract(sure_bg , sure_fg)

#creating barrier between bg & fg
_ , markers = cv2.connectedComponents(sure_fg)

# adding 1 to the background areas as we want unknowns to be 0
markers = markers+1
markers[unknown==255] =0

markers = cv2.watershed(frame , markers)
cv2.imshow("frame", frame)

frame[markers==-1] = [0,0,255]
cv2.imshow("frame2", frame)
cv2.imwrite("Leaf_output.jpg" , frame)

cv2.waitKey()
cv2.destroyAllWindows()


