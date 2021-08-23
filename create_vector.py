# pip install --upgrade imutils
# pip install opencv-python
# pip install cmake 
# pip install dlib
# update: foram mais montes de coisas, parei de escrever mas o Google ajudou a instalar o dlib

# Import the necessary packages
from scipy.spatial import distance as dist  # Para a euclidean distance entre landmarks
from imutils import face_utils              # Face processing functions
import argparse                             # To parse arguments
from imutils.video import VideoStream       # For real-time add this
import imutils                              # Image processing functions
import dlib                                 # Detect and localize landmks
import cv2                                  # Open CV
import time                                 # For real-time add this
from scipy import stats
import statistics
import matplotlib.pyplot as plt
import pickle
# Define the functions needed

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

def test(nome_video,frames_vector):
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
    EAR_CONSEC_FRAMES = 38 #48
    YAWN_CONSEC_FRAMES = 40 #60

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


    vs=cv2.VideoCapture("videos_i8/"+nome_video+".mp4")    
   
   
    #array with all values of lip_distance
    lip_array=[]
    #calculate the total number of frames
    num_frames=0
    #array with all values of ear
    ear_array=[]

    print(nome_video)
     # Loop over frames from the video stream (editable for videos from DROZY)
    for i in frames_vector:
        num_frames+=1
        vs.set(1,i)
        ret, frame = vs.read()  # For real-time edit to this
        if not ret:
            break
        # Resize frame
        frame = imutils.resize(frame, width=450)  # For real-time edit to this
        
        # Gray (eliminate the '3' for RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

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
            ear_array.append(ear)
           
            
           

            # Draw the computed EAR on the frame 
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (245, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            #-----YAWN-----
            
            # Compute the lip distance
            lip_distance = mouth_open(topLip, bottomLip)
            lip_array.append(lip_distance)
         
            
            # Draw the computed lip distance on the frame 
            cv2.putText(frame, "Lip Distance: {:.2f}".format(lip_distance), (245, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        # Visualize the frame
        cv2.imshow("Frame", frame)

        # Keyboard action!
        key = cv2.waitKey(1) & 0xFF 
        # If the 'q' key is pressed, break from the while loop
        if key == ord("q"):
            break


    # Do a bit of cleanings
    cv2.destroyAllWindows()
    cv2.waitKey(0)   # For real-time add this

   
   #normalize the vectors so the values go between 0 and 1
    ear_array_norm=[]
    for i in ear_array:
        ear_array_norm.append((i-min(ear_array))/(max(ear_array)-min(ear_array)))
    lip_array_norm=[]
    for i in lip_array:
        lip_array_norm.append((i-min(lip_array))/(max(lip_array)-min(lip_array)))
    
    
    print(num_frames)

    #if the ear go under 0.5 in ther normalize vector e add at the count
    ear_count=0
    for i in ear_array_norm:
        if i<0.5:
            ear_count+=1
    #if the lip distance 
    m=statistics.mode(lip_array)
    lip_count=0
    for i in lip_array:
        if i>m*6.5:
            lip_count+=1
    print("mode lip ", m)
 
    return [ear_count/num_frames,sum(ear_array_norm)/num_frames,lip_count/num_frames,sum(lip_array_norm)/num_frames]


class Model_drowsi:
    def __init__(self):
        self.loop= None
       

    def drowsi_model(self,type, nome_video):
        # Construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()        
        ap.add_argument("-p", "--shape-predictor", 
            default = "shape_predictor_68_face_landmarks.dat",               
            help = "path to facial landmark predictor") # ^^^^^
        ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system") # For real-time add this
        args = vars(ap.parse_args())

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
        if type==0:
                print("[INFO] accessing webcam...")              # For real-time add this
                vs = VideoStream(src=self.args["webcam"]).start()     # For real-time add this
                time.sleep(1.0)                                  # For real-time add this
        else:
                print(nome_video)
                vs=cv2.VideoCapture("videos_i8/"+nome_video+".mp4")                              # For real-time add this
                cv2.waitKey(0)
    
        lip_array=[]
            #calculate the total number of frames
        num_frames=0
            #array with all values of ear
        ear_array=[]
        while self.loop:
            
            if type==0:
                frame = vs.read()  # For real-time edit to this
            else:
                ret,frame = vs.read()
            
            if not ret:
                break
            # Gray (eliminate the '3' for RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

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
                ear_array.append(ear)
                #-----Checks BLINK-EAR-----
                
                # Check to see if the eye aspect ratio is below the blink threshold

                # Draw the computed EAR on the frame 
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (245, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                #-----YAWN-----
                
                # Compute the lip distance
                lip_distance = mouth_open(topLip, bottomLip)
                lip_array.append(lip_distance)
                #-----Checks YAWN-----
                
                
                # Draw the computed lip distance on the frame 
                cv2.putText(frame, "Lip Distance: {:.2f}".format(lip_distance), (245, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                 
                if num_frames==420:
                    result=self.check(ear_array, lip_array, num_frames)
                    filename = 'finalized_model.sav'
                    model = pickle.load(open(filename, 'rb'))
                    loaded_model = pickle.load(open(filename, 'rb'))
                    predictions=loaded_model.predict(result)
                    if result==2:
                         cv2.putText(frame, "LOW EAR ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif result==3:
                         cv2.putText(frame, "DROWSI ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    lip_array=[]
                    #calculate the total number of frames
                    num_frames=0
                        #array with all values of ear
                    ear_array=[]


                ret,buffer= cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()
                yield(b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'+frame +b'\r\n')


        cv2.destroyAllWindows()
        if type==0:
            vs.stop()
        else:
            cv2.waitKey(0)


    def check(self,ear_array,lip_array,num_frames):
    #normalize the vectors so the values go between 0 and 1
        ear_array_norm=[]
        for i in ear_array:
            ear_array_norm.append((i-min(ear_array))/(max(ear_array)-min(ear_array)))
        lip_array_norm=[]
        for i in lip_array:
            lip_array_norm.append((i-min(lip_array))/(max(lip_array)-min(lip_array)))


        #if the ear go under 0.5 in ther normalize vector e add at the count
        ear_count=0
        for i in ear_array_norm:
            if i<0.5:
                ear_count+=1
        #if the lip distance 
        m=statistics.mode(lip_array)
        lip_count=0
        for i in lip_array:
            if i>m*6.5:
                lip_count+=1
      
        return [ear_count/num_frames,sum(ear_array_norm)/num_frames,lip_count/num_frames,sum(lip_array_norm)/num_frames]