import cv2
import numpy as np 


#cap = cv2.VideoCapture(0)



image=cv2.pyrDown(cv2.imread("image.jpg"), cv2.IMREAD_UNCHANGED)
	#_ , frame = cap.read()
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
_ , thresh = cv2.threshold( gray , 127,255 , 0)
_ , contour , hier = cv2.findContours(thresh.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)


	#finding the bounding box coordinates
for c in contour:
	x,y,w,h = cv2.boundingRect(c)
	cv2.rectangle(image , (x,y) , (x+w , y+h) , (0,255,0) , 2)

	# finding minimum area 
	react = cv2.minAreaRect(c)

	# calculate the coordinate of the minimum area rectangle
	box = cv2.boxPoints(react)
	#normalise coordinates to integers
	box = np.int0(box)
	#draw contours
	cv2.drawContours(image , [box] , 0 , (0,255,0) ,3)

	# calculate center and radius of minimum enclosing circle
	(x,y),radius = cv2.minEnclosingCircle(c)
	centre = (int(x),int(y))
	radius = int(radius)

	img= cv2.circle(image , centre , radius , (0,0,255) ,2)

cv2.drawContours(image , contour , -1, (255,0,0) , 4)
	#cv2.imshow("contour" , img)
cv2.imshow("contours" , img)
cv2.imshow("thresh" , thresh)
cv2.imwrite("Contour.jpg", img)
	

cv2.waitKey()
	


#cap.release()
cv2.destroyAllWindows()