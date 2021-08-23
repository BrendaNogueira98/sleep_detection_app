# sleep_detection_app
This repository contais an app for driver drowsinees detection via eye monitoring being it closed or opened, and the same for mouth.


1.Install Python
2.Install Dlib
3.Install scipy
obs: maybe you need to put the python at the PATH of your computer

To run the app
1.Open the sleep_detection folder in your iDE 
2.write on the terminal "python facedetective.py"
3.then open the link that will appear at the terminal
4. to close the sever "ctrl c"

were implemented 3 diferent types of code:
1.the firts one were desenvolved by Telma Esteves, one of the supervisors of the work(drowsiness file)
2.the other were an adptation from https://github.com/fear-the-lord/Drowsiness-Detection (drowsinees_detection file)
3.And we trained a model that consider blinks anda mouth being opened, this model were trained with the files from videos_i8 (create_vector file)

You can choose a video to test: put it at the videos_i8 paste
You can also remove from comment of the second algorithm the cv2.imwrite function to save tha drowsiness frames in the past dataset




If you want to train the SVM model with more videos, go to the file Frames.xlsx and in the first column, put the name of the video in mp4, then the FPS are 30 or 15, and you can ignore with you want the column Frames. If  30 FPS,  the part of the video you choose must have a total of 840 frames, and if you have a 15 FPS, the part of the video must have a total of 420 frames. Put in the next column the frame of start and the frame of the end, and, finally, put the category where the part belongs. 
(1)If the person is awake
(2)If the person is lightly asleep
(3)If the person is drowsiness

To train run the "model.py", when it is over run the "model2.py". 
