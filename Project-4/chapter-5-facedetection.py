import cv2
import numpy as np 
import imutils
filepath ='C:\\Users\\ARPIT JAIN\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
cap = cv2.VideoCapture(0)

while True:


	_ , frame = cap.read()
	frame = imutils.resize(frame , width=500)
	gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier(filepath)
	eye_cascade = cv2.CascadeClassifier('\\\\'.join(filepath.split('\\')[:-1]) + '\\haarcascade_eye.xml')
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

	for (x,y,w,h) in faces:
		image = cv2.rectangle(frame.copy() , (x,y) , (x+w,y+h) , (0,255,0),2)

		roi = gray[y:y+h , x:x+w]
		eyes = eye_cascade.detectMultiScale(roi , 1.3,5 , 0 , (40,40))

		for (x_ , y_ , w_ , h_) in eyes:

			cv2.rectangle(image, (x_ , y_) , (x_ + w_ , y_+h_) , (0,0,255) , 2 )



	cv2.imshow("Original" , frame)
	cv2.imshow("facedetection", image)
	cv2.imwrite("Original.jpg" , frame)
	cv2.imwrite("facedetection.jpg", image)
	if cv2.waitKey(1)==32:
		break

cap.release()
cv2.destroyAllWindows()

