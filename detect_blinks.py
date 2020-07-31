#importing neccesary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream #accessing a file on disk
from imutils.video import VideoStream #accessing a live feed
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

#Eye blink detection with OpenCV, Python, and dlib

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
#ap.add_argument("-v", "--video", type=str, default="", help="path to input video file") #enable if video file wanted not live feed
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
#fileStream = True
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

"""
If you want to use a built-in webcam or USB camera, uncomment Line 56.
For a Raspberry Pi camera module, uncomment Line 57.
If you have uncommented either Line 56 or Line 57, then uncomment Line 64 as well to indicate that you are not reading a video 
file from disk.
"""

# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)