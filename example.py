import cv2
from gaze_tracking import GazeTracking
import numpy as np

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
moving_average = np.array([])
while True:
    # Exctract frame
    _, frame = webcam.read()

    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    moving_average = np.append(moving_average, int(gaze.is_center()))
    attention = np.mean(moving_average)*100
    text = f'Attention : {attention:.2f}%'
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
