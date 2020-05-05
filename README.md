  # Week 4
Install OpenCV on your system and go through the tutorial for OpenCV [here](https://www.youtube.com/playlist?list=PLvVx8lH-gGeC8XmmrsG855usswhwt5Tr1).

## Support Vector Machines

This week we'll be using Support Vecrtor machines to create a mini project of sorts. We'll be working on improving this project later on as well.
There are total four components in the file

* Main.py
 The main file for the program, here you will be loading in the SVM from sklearn and then train and store the model in a different file, which can be accessed by other files.
 You need to finish the SVM code in here
 
 * Detector.py
 The file will load in our image to test it on, and the classifier model, and predict the values.
 
 * photo8.jpg
 The sample File I'll be using to test the work. Once you've determined the model is working, you can swap it out for your own files to test it on.
 
 * digits_cls.pkl 
 The file you'll be generating by running Main.py to save the classifier, and the file Detector.py will  read to load in the classifier.

## Optional Exercise
Right now you're loading in data from one file, if you've gone through the openCV documentation,try and take the input from a livestream input via the webcam
