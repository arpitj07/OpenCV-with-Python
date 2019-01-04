import cv2
import numpy as np 
import csv

filename= 'C:\\Users\\ARPIT JAIN\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'

def generate():

	cap = cv2.VideoCapture(0)
	count=0

	while True:

		_ , frame = cap.read()
		gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

		face_cascade = cv2.CascadeClassifier(filename)
		#eye_cascade = cv2.CascadeClassifier('\\\\'.join(filename.split("\\")[:-1] )+ 'haarcascade_eye.xml')

		faces = face_cascade.detectMultiScale(gray , 1.3 , 5)

		for (x,y,w,h) in faces:

			cv2.rectangle(frame , (x,y),(x+w , y+h) , (0,0,255), 2)
			face = cv2.resize(gray[y:y+h ,x:x+w] , (200,200))
			file ="./datasets/image{}.pgm".format(count)
			cv2.imwrite(file , face)
			count+=1

		cv2.imshow("camera" , frame)
		if cv2.waitKey(1) ==32:
			break
	
	#writing CSV file
	with open("dataset.csv" , "w") as csvfile:
				filewriter = csv.writer(csvfile, delimiter=';',
                             quoting=csv.QUOTE_MINIMAL)
				for i in range(count):
					filewriter.writerow(["./datasets/image{}.pgm".format(i) ,0])
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	generate()