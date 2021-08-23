from scipy.spatial import distance as dist # Para a euclidean distance entre landmarks
from imutils import face_utils              # Face processing functions
import argparse                             # To parse arguments
from imutils.video import VideoStream       # For real-time add this
import imutils                              # Image processing functions
import dlib                                 # Detect and localize landmks
import cv2                                  # Open CV
import time  
from equalizer import *
def eye_aspect_ratio(eye):
    """returns the EAR (eye aspect ratio)"""
    
	# Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])

	# Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    h = dist.euclidean(eye[0], eye[3])

	# Compute the eye aspect ratio
    ear = (v1 + v2) / (2.0 * h)

	# Return the eye aspect ratio
    return ear
def mouth_open(topLip, bottomLip):
    """returns the lip distance"""
    
	# Find the lip center - maybe improvement: lip average
    topLipCenter = topLip[2]
    bottomLipCenter = bottomLip[2]

	# Compute the euclidean distance between the top and bottom lip centers (x, y)-coordinates
    lip_distance = dist.euclidean(topLipCenter, bottomLipCenter)

    return lip_distance
class Video:
    def __init__(self):
        self.loop= None
    def drowsi(self,type,nome_video):
        # Construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()        
        ap.add_argument("-p", "--shape-predictor", 
            default = "shape_predictor_68_face_landmarks.dat",               
            help = "path to facial landmark predictor") # ^^^^^
        ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system") # For real-time add this
        args = vars(ap.parse_args())

        # Define a constant to indicate blink (threshold) - VARIES FROM SUBJECT TO SUBJECT
        EAR_THRESH = 0.25
        YAWN_THRESH = 15

        # Define a constant for the number of consecutive frames the eye/lip-dist must be low/high to raise alert
        EAR_CONSEC_FRAMES = 38
        YAWN_CONSEC_FRAMES = 40

        # Initialize the frame counter
        bCOUNTER = 0 # Blink-EAR
        yCOUNTER = 0 # Yawn

        # Initialize dlib's face detector (HOG-based)
        print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
                    
        # Create the facial 68-landmark predictor
        predictor = dlib.shape_predictor(args["shape_predictor"])

        #-----LANDMARKS INDEXES for different face parts-----
        # Grab the indexes of the facial landmarks for the left and right eyes
        (lEyeStart, lEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rEyeStart, rEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # Grab the indexes of the facial landmarks for the inner mouth
        (iMouthStart, iMouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

        # Get frames from videostream (webcam)
        if type==0:
            print("[INFO] accessing webcam...")              # For real-time add this
            vs = VideoStream(src=args["webcam"]).start()     # For real-time add this
            time.sleep(1.0)                                  # For real-time add this
        else:
            print(nome_video)
            vs=cv2.VideoCapture("videos_i8/"+nome_video+".mp4")                              # For real-time add this
            cv2.waitKey(0)
        # Loop over frames from the video stream (editable for videos from DROZY)
        while (self.loop):
            if type==0:
                frame = vs.read()  # For real-time edit to this
            else:
                ret,frame = vs.read()
            # Resize frame
           # frame = imutils.resize(frame, width=450)  # For real-time edit to this
            
            # Gray (eliminate the '3' for RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

            #applying histogram equalization
            gray_frame=equalizer(gray_frame)
            # Detect faces in the grayscale frame
            rects = detector(gray_frame, 0)

            # Loop over the face detections
            for rect in rects:
                
                # Determine the 68 facial landmarks for the face region
                shape = predictor(gray_frame, rect)
                
                # Convert the facial landmark (x, y)-coordinates to a numpy array
                shape = face_utils.shape_to_np(shape)

                #EYES-----
                    
                # Extract the left and right eye coordinates
                leftEye = shape[lEyeStart:lEyeEnd]
                rightEye = shape[rEyeStart:rEyeEnd]
                
                # Compute the convex hull for the left and right eye
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                
                # Visualize both eyes on the frame        
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                
                #MOUTH-----
                    
                # Extract the inner lips coordinates
                topLip = shape[60:65]
                bottomLip = shape[64:68]
                
                # Extract the inner mouth coordinates
                innerMouth = shape[iMouthStart:iMouthEnd]
                
                # Compute the convex hull for the inner mouth
                innerMouthHull = cv2.convexHull(innerMouth)
                
                # Visualize mouth on the frame        
                cv2.drawContours(frame, [innerMouth], -1, (0, 255, 0), 1)
                            
                #-----BLINK-EAR-----
                
                # Compute the eye aspect ratio for both eyes using both eye coordinates
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # Average the eye aspect ratio for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                #-----Checks BLINK-EAR-----
                
                # Check to see if the eye aspect ratio is below the blink threshold
                if ear < EAR_THRESH:
                    # If so, increment the blink frame counter
                    bCOUNTER += 1
                    # If so, check to see if the eyes were closed for a sufficient number of frames
                    if bCOUNTER >= EAR_CONSEC_FRAMES:                    
                        # If so, draw a warning on the frame
                        cv2.putText(frame, "LOW EAR ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Otherwise, the eye aspect ratio is not below the blink threshold
                else:
                    # Reset the frame counter 
                    bCOUNTER = 0

                # Draw the computed EAR on the frame 
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (245, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                #-----YAWN-----
                
                # Compute the lip distance
                lip_distance = mouth_open(topLip, bottomLip)
                
                #-----Checks YAWN-----
                
                # Check to see if the lip distance is above the yawn threshold
                if lip_distance > YAWN_THRESH:
                    # If so, increment the yawn frame counter
                    yCOUNTER += 1
                    # If so, check to see if the lips were sufficiently parted for a sufficient number of frames
                    if yCOUNTER >= YAWN_CONSEC_FRAMES:                    
                        # If so, draw a warning on the frame
                        cv2.putText(frame, "YAWN ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Otherwise, the lip distance is not below the yawn threshold
                else:
                    # Reset the frame counter 
                    yCOUNTER = 0

                # Draw the computed lip distance on the frame 
                cv2.putText(frame, "Lip Distance: {:.2f}".format(lip_distance), (245, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            
            # Visualize the frame
            #cv2.imshow("Frame", frame)


            ret,buffer= cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'+frame +b'\r\n')



            # Keyboard action!
            key = cv2.waitKey(1) & 0xFF 
            # If the 'q' key is pressed, break from the while loop
            if key == ord("q"):
                break


        # Do a bit of cleaning
        cv2.destroyAllWindows()
        if type==0:
            vs.stop()
        else:
            cv2.waitKey(0)  # For real-time add this