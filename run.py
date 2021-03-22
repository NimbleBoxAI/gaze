import cv2
from gaze_tracking import GazeTracking
import numpy as np

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
moving_average = np.array([1])
window = np.array([])
attention = 100.00
count = 0
SLIDER = 30
THRESHOLD = 0.1
while True:
    # Exctract frame
    _, frame = webcam.read()
    count+= 1
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    window = np.append(window, int(gaze.is_center()))
    if count%SLIDER == 0:
        moving_average = np.append(moving_average, int(np.mean(window[-SLIDER:])>THRESHOLD))
        attention = np.mean(moving_average)*100
        window = np.array([])
    text = f'Attention : {attention:.2f}%'
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    cv2.imshow("Demo", frame)
    if cv2.waitKey(1) == 27:
        break
