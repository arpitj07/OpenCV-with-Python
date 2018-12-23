import cv2
import numpy as np 


def sharpFilter():
	sharpkernel = np.array([[-1,-1,-1],
							[-1,9,-1],
							[-1,-1,-1]])
	return sharpkernel

def edgeFilter():
	edgekernel = np.array([[-1,-1,-1],
							[-1,8,-1],
							[-1,-1,-1]])
	return edgekernel

def blurFilter():
	blurkernel = np.ones([5,5]) * 0.04
	return blurkernel

def embossFilter():
	embosskernel = np.array([[-2,-1,0],
							[-1,1,1],
							[0,1,2]])
	return embosskernel

def applyFilter(src ,dst, function_name):
	return cv2.filter2D(src , -1 , function_name , dst)


cap = cv2.VideoCapture(0)


while True:
	_, frame = cap.read()

	dst = frame.shape[:2]
	dst1 = frame.shape[:2]
	dst2 = frame.shape[:2]
	dst3 = frame.shape[:2]

	frame1= applyFilter(frame , dst,sharpFilter())
	frame2= applyFilter(frame , dst1,edgeFilter())
	frame3= applyFilter(frame , dst2,blurFilter())
	frame4= applyFilter(frame , dst3,embossFilter())


	cv2.imshow("original", frame)
	cv2.imshow("sharp" , frame1)
	cv2.imshow("edge " , frame2)
	cv2.imshow("blur" , frame3)
	cv2.imshow("emboss", frame4)

	if cv2.waitKey(1)== 32:
		break

cv2.destroyAllWindows()
cap.release()
