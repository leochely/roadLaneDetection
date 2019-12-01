import numpy as np
import cv2
import functions

# Video input
cap = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

# Video output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

# Processing video
while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


cap.release()
cv2.destroyAllWindows()
