import os
import sys
import cv2
import numpy as np
from load_data import load_images
from Normalize import normalize


if __name__ == "__main__":
    

    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    names = ['Arpit', 'Jane', 'Jack']
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    if len(sys.argv) < 2:
        print( "USAGE: facerec_demo.py </path/to/images> [</path/to/store/images/at>]")
        sys.exit()
    # Now read in the image data. This must be a valid path!
    [X,y] = load_images(sys.argv[1])
    # Convert labels to 32bit integers. This is a workaround for 64bit machines,
    # because the labels will truncated else. This will be fixed in code as
    # soon as possible, so Python users don't need to know about this.
    # Thanks to Leo Dirac for reporting:
    y = np.array(y, dtype=np.int32)
    # If a out_dir is given, set it:
    if len(sys.argv) == 3:
        out_dir = sys.argv[2]
    # Create the Eigenfaces model. We are going to use the default
    # parameters for this simple example, please read the documentation
    # for thresholding:
    #model = cv2.face.createLBPHFaceRecognizer()
    model = cv2.face.EigenFaceRecognizer_create()
    # Read
    # Learn the model. Remember our function returns Python lists,
    # so we use np.asarray to turn them into NumPy lists to make
    # the OpenCV wrapper happy:
    model.train(np.asarray(X), np.asarray(y))
    # We now get a prediction from the model! In reality you
    # should always use unseen images for testing your model.
    # But so many people were confused, when I sliced an image
    # off in the C++ version, so I am just using an image we
    # have trained with.
    #
    # model.predict is going to return the predicted label and
    # the associated confidence:
    

    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    

    while (True):
      read, img = camera.read()
      faces = face_cascade.detectMultiScale(img, 1.3, 5)
      

      for (x, y, w, h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi = gray[x:x+w, y:y+h]
        
        roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
        print(roi.shape)
        params = model.predict(roi)
        print( "Label: {}, Confidence: {}".format(params[0], params[1]))
        cv2.putText(img, names[params[0]], (x,y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
      cv2.imshow("camera", img)
      if cv2.waitKey(1)==32:
        break

    [p_label, p_confidence] = model.predict(np.asarray(X[0]))
    # Print it:
    print( "Predicted label = {} (confidence={})".format(p_label, p_confidence))
    # Cool! Finally we'll plot the Eigenfaces, because that's
    # what most people read in the papers are keen to see.
    
    # Just like in C++ you have access to all model internal
    # data, because the cv::FaceRecognizer is a cv::Algorithm.

    # You can see the available parameters with getParams():
    #print( model.getParams())
    # Now let's get some data:
    mean = model.getMean()
    eigenvectors = model.getEigenVectors()


    # We'll save the mean, by first normalizing it:
    mean_norm = normalize(mean, 0, 255, dtype=np.uint8)
    mean_resized = mean_norm.reshape(X[0].shape)
    
    if out_dir is None:
        cv2.imshow("mean", mean_resized)
    else:
        cv2.imwrite("{}/mean.png".format(out_dir), mean_resized)
    # Turn the first (at most) 16 eigenvectors into grayscale
    # images. You could also use cv::normalize here, but sticking
    # to NumPy is much easier for now.
    # Note: eigenvectors are stored by column:
    
    for i in range(min(len(X), 16)):
        eigenvector_i = eigenvectors[:,i].reshape(X[0].shape)
        eigenvector_i_norm = normalize(eigenvector_i, 0, 255, dtype=np.uint8)
        # Show or save the images:
        if out_dir is None:
            cv2.imshow("{}/eigenface_{}".format(out_dir,i), eigenvector_i_norm)
        else:
            cv2.imwrite("{}/eigenface_{}.png".format(out_dir,i), eigenvector_i_norm)
    # Show the images:
    if out_dir is None:
        cv2.waitKey(0)

    cv2.destroyAllWindows()