import numpy as np
import cv2
import functions

# Input and Output File Names
input_video_file_name = 'Lane Detection Test Video 01.avi'

# Video input
cap = cv2.VideoCapture(input_video_file_name)

# Check if input video opened correctly
if(cap.isOpened() == False):
    print("Unable to read input video file")

# Convert input resolutions from float to integer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Find total number of frames
amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#print("Total number of frames: ", amount_of_frames)

# Frame counter
count = 0

# ---------------------------- RIGHT HERE -------------------------------------------------
# Desired Frame
desired_frame = 50
#------------------------------------------------------------------------------------------

# Check desired frame against total number of frames
if desired_frame > amount_of_frames:
    desired_frame = amount_of_frames

while (True):
    count = count + 1

    ret, frame = cap.read()

    # If there is a frame
    if ret == True:

        if count == desired_frame:

            # Find lane
            lane_pts = functions.findLaneLines(frame)
            img_with_overlay = functions.overlay(frame, lane_pts)

            # Display resulting img
            functions.ShowImage('Returned img', img_with_overlay)

    # If NO frame
    else:
        # Break the loop
        break

# Release the input and output video objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
