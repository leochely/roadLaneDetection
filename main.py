import numpy as np
import cv2
import functions

# Video input
cap = cv2.VideoCapture('Lane Detection Test Video 01.avi')

# Video output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280,720))

# Processing video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        # Writing original frame with the highlighted lines to output
        output.write(gray)
    else:
        break

cap.release()
output.release()
cv2.destroyAllWindows()
