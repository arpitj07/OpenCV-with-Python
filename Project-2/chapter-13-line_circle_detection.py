import cv2
import numpy as np 


def lineDetection():

	image = cv2.imread("image6.jpg")
	gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
	_ , thresh = cv2.threshold(gray , 127 , 255 , 0)
	edges = cv2.Canny(gray , 50 , 150, apertureSize=3 , L2gradient=False)
	minLineLength = 100
	maxLineGap = 10

	lines = cv2.HoughLinesP(edges ,1, np.pi/180 , 100, minLineLength , maxLineGap )

	print(lines[0])

	for x1,y1,x2,y2 in lines[0]:
		
		cv2.line(image,(x1,y1),(x2,y2),(0,0,255),3)

	cv2.imshow("Linedetection" , image)
	cv2.imwrite("Linedetection.jpg", image)
	cv2.imshow("Thresh" , edges)
	cv2.waitKey(5000)

def lineDetection2():

	img = cv2.imread('image6.jpg')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)

	lines = cv2.HoughLines(edges,1,np.pi/180,200)
	for rho,theta in lines[0]:
	    a = np.cos(theta)
	    b = np.sin(theta)
	    x0 = a*rho
	    y0 = b*rho
	    x1 = int(x0 + 1000*(-b))
	    y1 = int(y0 + 1000*(a))
	    x2 = int(x0 - 1000*(-b))
	    y2 = int(y0 - 1000*(a))


	cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

	cv2.imshow("Linedetection" , img)
	cv2.imwrite("Linedetection.jpg", img)
	cv2.imshow("Thresh" , edges)
	cv2.waitKey(5000)


def circleDetection():
	image = cv2.imread("images5.jpg")
	gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
	blur = cv2.medianBlur(gray , 5)

	circles = cv2.HoughCircles(blur , cv2.HOUGH_GRADIENT , 1,120 , 
				   param1=100 , param2=30 , minRadius=0 , maxRadius=0)

	circles = np.uint16(np.around(circles))

	for i in circles[0,:]:
		#draw the outer circle
		cv2.circle(image , (i[0], i[1]), i[2] , (0,255,0),2)

		# draw the centre circle
		cv2.circle(image , (i[0],i[1]) , 2 , (0,0,255), 3)


	cv2.imshow("Circledetection" , image)
	cv2.imwrite("Circledetection.jpg", image)
	cv2.imshow("Thresh" , blur)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()


def main():

	answer = input("enter key:" )
	if answer=="L":
		lineDetection()
	elif answer=="l":
		lineDetection2() 
	elif answer=="C" or answer=="c":
		circleDetection()
	
	

if __name__ =="__main__":

	main()
