import numpy as np
import cv2

cap = cv2.VideoCapture('Lane Detection Test Video 01.mp4'

while(capi.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


cap.release()
cv2.destroyAllWindows()
