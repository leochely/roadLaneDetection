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
desired_frame = 1
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
            # Find Lane Lines
            height, width = frame.shape[0:2]
            region_of_interest_vertices = [
                (0, height),
                (width / 2, height / 2),
                (width, height),
            ]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Canny Edge Detection
            cannyed_image = cv2.Canny(gray, 100, 200)
            print(cannyed_image.shape)
            cropped_image = functions.region_of_interest(
                cannyed_image,
                np.array([region_of_interest_vertices], np.int32))

            # Hough Lines
            lines = cv2.HoughLinesP(
                cropped_image,
                rho=6,
                theta=np.pi / 60,
                threshold=160,
                lines=np.array([]),
                minLineLength=40,
                maxLineGap=25
            )

            # Write original frame with the highlighted lines to output
            color = [255, 0, 0]
            thickness = 3
            line_image = np.zeros(
                    (
                        frame.shape[0],
                        frame.shape[1],
                        3
                    ),
                    dtype=np.uint8,
                )
            # Loop over all lines and draw them on the blank image.
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

            # Merge the image with the lines onto the original.
            img = cv2.addWeighted(frame, 0.8, line_image, 1.0, 0.0)

            # Display resulting image
            #functions.ShowImage('Returned img', img_with_overlay)
            cv2.imwrite('temp.png', img)

    # If NO frame
    else:
        # Break the loop
        break

# Release the input and output video objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
