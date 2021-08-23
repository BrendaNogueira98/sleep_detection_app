# Import the necessary packages 
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from EAR_calculator import *
from imutils import face_utils 
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from matplotlib import style 
import imutils 
import dlib
import time 
import argparse 
import cv2 
from playsound import playsound
from scipy.spatial import distance as dist
import os 
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from equalizer import *

#code from https://github.com/fear-the-lord/Drowsiness-Detection with a few alterations 
style.use('fivethirtyeight')
# Creating the dataset 
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
class Video_2():
    def __init__(self):
        self.loop_2= None
    def detect(self,type,nome_video):
        
        style.use('fivethirtyeight')
        
        ap = argparse.ArgumentParser() 
        ap.add_argument("-p", "--shape_predictor", required = False,default = "shape_predictor_68_face_landmarks.dat", help = "path to dlib's facial landmark predictor")
        ap.add_argument("-r", "--picamera", type = int, default = 0, help = "whether raspberry pi camera shall be used or not")
        args = vars(ap.parse_args())

        # Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink 
        EAR_THRESHOLD = 0.2 #0.3
        # Declare another costant to hold the consecutive number of frames to consider for a blink 
        CONSECUTIVE_FRAMES = 15 #20
        # Another constant which will work as a threshold for MAR value
        MAR_THRESHOLD = 24 #14
        #Constant to limit the blinks frames
        BLINK_FRAMES=14
        #Constant to count the number of consecutive frames to consider the person is awake
        EYES_OPEN=80
        #Constant to define the consecutive frames the mouth is open
        YAWN_FRAMES=14

        # Initialize counters 
        BLINK_COUNT = 0 
        FRAME_COUNT = 0 
    
        

        # Now, intialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
        print("[INFO]Loading the predictor.....")
        detector = dlib.get_frontal_face_detector() 
        predictor = dlib.shape_predictor(args["shape_predictor"])

        # Grab the indexes of the facial landamarks for the left and right eye respectively 
        (lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    
        if(type==0):
            # Now start the video stream and allow the camera to warm-up
            print("[INFO]Loading Camera.....")
            vs = VideoStream(usePiCamera = args["picamera"] > 0).start()
            time.sleep(1) 
        else:
            # Now start the video 
            print(nome_video)
            vs=cv2.VideoCapture("videos_i8/"+nome_video+".mp4")                              # For real-time add this
           # cv2.waitKey(0)
    
        assure_path_exists("dataset/")
        #iniciate counts to save the frames
        count_sleep = 0
        count_yawn = 0 

        #iniciate the counts of yawn and the time the eyws are open
        count_time_yawn=0
        count_time_eyes_open=0

        
        # Now, loop over all the frames and detect the faces
        while self.loop_2: 
            # Extract a frame 
            if type==0:
                frame = vs.read()  # For real-time edit to this
            else:
                ret,frame = vs.read()
           
           
            # Convert the frame to grayscale 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

             #applying histogram equalization
            gray=equalizer(gray)
            # Detect faces 
            rects = detector(frame, 1)

            # Now loop over all the face detections and apply the predictor 
            for (i, rect) in enumerate(rects): 
                shape = predictor(gray, rect)
                # Convert it to a (68, 2) size numpy array 
                shape = face_utils.shape_to_np(shape)

                # Draw a rectangle over the detected face 
                (x, y, w, h) = face_utils.rect_to_bb(rect) 
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)	
                # Put a number 
                cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                leftEye = shape[lstart:lend]
                rightEye = shape[rstart:rend] 
                mouth = shape[mstart:mend]
                # Compute the EAR for both the eyes 
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # Take the average of both the EAR
                EAR = (leftEAR + rightEAR) / 2.0
                
                
                # Compute the convex hull for both the eyes and then visualize it
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                # Draw the contours 
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

                MAR = mouth_aspect_ratio(mouth)
               
                # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
                # Thus, count the number of frames for which the eye remains closed 
                if EAR < EAR_THRESHOLD: 
                    FRAME_COUNT += 1
                    BLINK_COUNT+=1

                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

                    if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
                        count_sleep += 1
                        # Add the frame to the dataset ar a proof of drowsy driving
                        #cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
                        playsound('alarm.mp3')
                        cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else: 
                    #check the blinks

                    #if the eye is open for a very long time turn the blink count into 0
                    count_time_eyes_open+=1
                    if(count_time_eyes_open>EYES_OPEN):
                        BLINK_COUNT=0

                    #if the eye is open but the person already blinked a lot
                    if BLINK_COUNT >= BLINK_FRAMES: 
                       # play a beep to get the person's attention
                       #playsound('beep.mp3')

                       BLINK_COUNT = 0
                       #blink alarm
                       cv2.putText(frame, "BLINK ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    FRAME_COUNT = 0
                cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Check if the person is yawning
                if MAR > MAR_THRESHOLD:
                    count_yawn += 1
                    count_time_yawn +=1
                    cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
                    if(count_time_yawn>=YAWN_FRAMES) :
                        #show a yanw alert
                        cv2.putText(frame, "Yawn ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # Add the frame to the dataset ar a proof of drowsy driving
                       # cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)
                        # play a beep to get the person's attention
                        playsound('beep.mp3')

            #display the frame 
            ret,buffer= cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'+frame +b'\r\n')
       
       
       
        cv2.destroyAllWindows()
        if type==0:
            vs.stop()
        else:
            cv2.waitKey(0)
