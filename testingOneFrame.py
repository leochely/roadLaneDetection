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

# ---------------------------- CHANGE THIS HERE -------------------------------------------------
# Desired Frame
desired_frame = 100
#------------------------------------------------------------------------------------------

# Check desired frame against total number of frames
if desired_frame > amount_of_frames:
    desired_frame = amount_of_frames

# For visualization
overlayROI = False

while (True):
    # Frame counter
    count = count + 1

    ret, frame = cap.read()

    # If there is a frame
    if ret == True:

        if count == desired_frame:

            if overlayROI:  # Show ROI Overlay (cannot do this AND find lanes in same run)
                # In the order of top left, top right, bottom right, bottom left
                roi_pts = np.array([[590, 530], [682, 530], [1100, 660], [124, 660]], np.int32)

                # Show region of interest
                roi = functions.overlayROI(frame, roi_pts)
                functions.ShowImage("RegionOfInterest", roi)
                #cv2.imwrite('Output_Images/ROI.jpg', roi)

            else:       # Find Lanes
                img_with_overlay = functions.findLaneLines(frame)

                # Display resulting image
                functions.ShowImage('Returned img', img_with_overlay)

    # If NO frame
    else:
        # Break the loop
        break

# Release the input and output video objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
