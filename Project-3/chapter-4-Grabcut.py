import cv2
import numpy as np 
import matplotlib.pyplot as plt 

#cap = cv2.VideoCapture(0)


bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

rect = (100 , 51, 471 , 378)
iteration =5

while True:

	#ret , frame = cap.read()
	frame = cv2.imread("image1.jpg")
	mask = np.zeros(frame.shape[:2], np.uint8)
	x,y,w,h = rect
	#cv2.rectangle(frame , (x,y) , (w,h) , (0,255, 0) ,2)
	cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,iteration,cv2.GC_INIT_WITH_RECT)

	mask2 = np.where((mask==2)| (mask==0),0,1).astype("uint8")

	image = frame*mask2[:,:,np.newaxis]

	#plt.subplot(121)
	#lt.imshow(image)
	#plt.title("grabCut")
	#plt.xticks(),plt.yticks()

	#plt.subplot(122)
	#plt.imshow(frame)
	#plt.title("original")
	#plt.xticks(),plt.yticks()

	cv2.imshow("grabCut" , image)
	cv2.imshow("original" , frame )
	cv2.imwrite("grabCut.jpg" , image)
	cv2.imwrite("original.jpg" , frame)


	if cv2.waitKey(1)==32:
		break

#cap.release()
cv2.destroyAllWindows()