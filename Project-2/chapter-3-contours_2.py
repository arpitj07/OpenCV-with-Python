import cv2
import numpy as np 



image=cv2.pyrDown(cv2.imread("image3.jpg"), cv2.IMREAD_UNCHANGED)
#blurred = cv2.pyrMeanShiftFiltering(image , 31,91)
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
_ , thresh = cv2.threshold( gray , 127,255 , 0)
_ , contour , hier = cv2.findContours(thresh.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)


for cnt in contour:
	epsilon = 0.01*cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt , epsilon , True)
	
	hull = cv2.convexHull(cnt)

image1=cv2.drawContours(image.copy(),[approx],-1,(0,0,255),2)
image2=cv2.drawContours(image.copy() , [hull] , -1 , (0,255,0),2)

cv2.imshow("approx contour", image1)
cv2.imshow("thresh",thresh)
cv2.imshow("convexHull",image2)
cv2.imshow("Original", image)
cv2.imwrite("ConvexHull.jpg", image2)
cv2.imwrite("approxContour.jpg", image1)
cv2.waitKey()
		
#cap.release()		
cv2.destroyAllWindows()