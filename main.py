'''
main.py file for use in Final Project of CSCI437 - Fall 2019

Authors: Curtis Burke, Leo Chely
Assignment: Final Project
Course: CSCI437
Date: December 8, 2019

'''

# Import libraries
import numpy as np
import cv2
import functions

# Input and Output File Names
input_video_file_name = 'Lane Detection Test Video 01.avi'
output_video_file_name = 'BurkeChely_CSCI437_FinalProject_output.avi'

# Video input
cap = cv2.VideoCapture(input_video_file_name)

# Check if input video opened correctly
if(cap.isOpened() == False):
    print("Unable to read input video file")

# Convert input resolutions from float to integer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Video output
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
output = cv2.VideoWriter(output_video_file_name, fourcc, 30, (frame_width, frame_height))

# Process video
while (True):
    # Read current frame
    ret, frame = cap.read()

    # If there is a frame
    if ret == True:

        # Find lane lines
        frame_with_overlay = functions.findLaneLines(frame)

        # Write original frame with the highlighted lines to output
        output.write(frame_with_overlay)

        # Display resulting frame (for visualization)
        cv2.imshow('frame', frame_with_overlay)

        # Press Q on keyboard to stop playing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # If NO frame
    else:
        # Break the loop
        break

# Release the input and output video objects
cap.release()
output.release()

# Closes all the frames
cv2.destroyAllWindows()
