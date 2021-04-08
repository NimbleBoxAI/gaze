import cv2
import numpy as np
from gaze_tracking import GazeTracker
from expression.src.predict import predict_emotion
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Object creation for gaze tracking.
gaze = GazeTracker()

# Initializing the videoCapture object to infer the webcam stream.
webcam = cv2.VideoCapture(0)
moving_average = np.array([1])
window = np.array([])

# Defining constant that affect the way gaze tracking is done. 
attention = 100.00
count = 0
SLIDER = 30
THRESHOLD = 0.15
while True:

    # Exctracting frame from the VideoCapture object.
    _, frame = webcam.read()
    count+= 1
    emotion, face_found = predict_emotion(frame)

    # The frame to infer is sent to the infer function here. 
    gaze.infer(frame)
    
    window = np.append(window, int(gaze.is_center()))

    # Here the moving average is calculated.
    if count%SLIDER == 0:
        moving_average = np.append(moving_average, int(np.mean(window[-SLIDER:])>THRESHOLD))
        attention = np.mean(moving_average)*100
        window = np.array([])

    # Displaying the attention metric and infered frame to the screen.    
    text = f'Attention : {attention:.2f} % '
    if face_found:
      text2 = f'Emotion : {emotion}'
    else:
      text2 = 'No face found'
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    cv2.putText(frame, text2, (90, 120), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    cv2.imshow("Demo", frame)
    if cv2.waitKey(1) == 27:
        break
