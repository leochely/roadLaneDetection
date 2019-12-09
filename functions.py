'''
function.py file for use with main.py in Final Project of CSCI437 - Fall 2019

Authors: Curtis Burke, Leo Chely
Assignment: Final Project
Course: CSCI437
Date: December 8, 2019

'''

import numpy as np
import math
import cv2

def findLaneLines(frame):
    '''
    Lane Line finding algorithm

    inputs: frame - unedited frame image from input video

    outputs: image - combined image of original frame and processed frame with lane line overlayed

    '''

    #cv2.imwrite('Output_Images/frame.jpg', frame)      # For use in Report

    # ROI points - In the order of top left, top right, bottom right, bottom left
    src_pts = np.float32(
        [
            [520.0, 560.0],
            [740.0, 560.0],
            [1000.0, 700.0],
            [270.0, 700.0],
        ]
    )

    # Warp lane in front of car to bird-eye view
    warped_img, M, Minv = warp_image(frame, src_pts)
    img = warped_img.copy()
    #ShowImage("Warped Image", warped_img)               # For visualization
    #cv2.imwrite('Output_Images/warpped.jpg', img)       # For use in Report

    # ROI in Warped Image
    roi_pts = np.array([[200, 0], [1000, 0], [1000, 720], [200, 720]], np.int32)


    # Threshold warped image (grayscale, and binary)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #cv2.imwrite('Output_Images/binary.jpg', img)       # For use in Report

    # Canny Edge Detection
    img = cv2.Canny(img, 100, 200)
    #ShowImage('Canny Edge Image', img)               # For visualization
    #cv2.imwrite('Output_Images/canny.jpg', img)      # For use in Report

    # Apply Gaussian Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Create mask for ROI
    mask = np.zeros_like(img)
    # Fill mask
    cv2.fillPoly(mask, [roi_pts], 255)
    # Join mask and img
    masked_img = cv2.bitwise_and(img, mask)
    #ShowImage('Masked Image (ROI)', masked_img)     # For visualization
    img = masked_img.copy()

    # Hough Lines         edges, rho   theta   thresh          min length, max gap
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, np.array([]), 100, 250)

    # Draw Hough Lines
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 1)
    #ShowImage("Hough Lines", img)                  # For visualization
    #cv2.imwrite('Output_Images/hough.jpg', img)    # For use in Report

    # Convert Hough lines for a single Left and Right Line
    left_line, right_line = houghLinestoLaneLines(img, lines)

    # Overlay on Warped Img
    overlayed_img = overlay(warped_img, left_line)
    overlayed_img = overlay(warped_img, right_line)
    #ShowImage("Overlayed on Warped", overlayed_img)                        # For visualization
    #cv2.imwrite('Output_Images/overlayed_warpped.jpg', overlayed_img)      # For use in Report

    # Unwarp Birds-eye View (M_inverse)
    unWarped_img = cv2.warpPerspective(warped_img, Minv, (frame.shape[1], frame.shape[0]))
    #cv2.imwrite('Output_Images/unwarpepd.jpg', unWarped_img)               # For use in Report

    # Create Region of interest for masking
    unWarped_roi = np.array([src_pts[0], src_pts[1], src_pts[2], src_pts[3]], np.int32)

    # Combine frame and unWarped image with a mask around the ROI
    combinded_img = combineImages(frame, unWarped_img, unWarped_roi)
    #ShowImage("Comb", combinded_img)                                       # For visualization
    #cv2.imwrite('Output_Images/combined.jpg', combinded_img)               # For use in Report

    return combinded_img


def combineImages(img1, img2, roi):

    # Create Mask
    mask = np.zeros_like(img1)
    # Fill mask
    cv2.fillPoly(mask, [roi], (255, 255, 255))
    # Convert mask to binary (for adding)
    mask = np.uint8(mask)
    # Invert Mask
    mask_inv = cv2.bitwise_not(mask)

    # Add images together with masks
    temp = cv2.bitwise_and(img2, mask)
    temp2 = cv2.bitwise_and(img1, mask_inv)

    # Add temp images together
    combined_img = cv2.add(temp, temp2)

    return combined_img

def houghLinestoLaneLines(img, lines):

    # Initialize vectors to hold needed information
    left_lines = []         # (slope, intercept)
    left_weights = []       # length
    right_lines = []
    right_weights = []

    width, _ = img.shape

    # Iterate through all lines
    for line in lines:
        # Extract points from this line
        for x1, y1, x2, y2 in line:

            # Check for infinite demoninator (Vertical line)
            if (x2 - x1) == 0:
                continue

            # Calculate Slope
            slope = (y2 - y1) / (x2 - x1)

            # Only keep extreeme values
            if math.fabs(slope) < 1:
                continue

            # Calculate intercept and length
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            # Place points in vectors based on slope and region of image
            if slope < 0 and x1 < 3 * width / 4:  # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))

            elif slope > 0 and x1 > width / 4:
                right_lines.append((slope, intercept))
                right_weights.append((length))

            else:
                continue

    # Preform Weighted Averaging
    if len(left_weights) > 0:
        left_line = np.dot(left_weights, left_lines) / np.sum(left_weights)
    else:
        left_line = None

    if len(right_weights) > 0:
        right_line = np.dot(right_weights, right_lines) / np.sum(right_weights)
    else:
        right_line = None

    return left_line, right_line


def warp_image(img, src_pts):
    # Get image shape
    height, width = img.shape[0:2]

    # In the order of top left, top right, bottom right, bottom left
    src = src_pts
    dst = np.float32(
        [
            [200.0, 0],
            [width - 200.0, 0],
            [width - 200.0, height],
            [200.0, height],
        ]
    )           # For use with this particular birds-eye ROI

    # Get transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp
    warped_img = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)

    return warped_img, M, Minv

def overlay(img, line):

    # Check if line exists
    if line is None:
        return img

    # Collect image shape
    y1 = img.shape[0]
    y2 = 0

    # Extact line information
    slope, intercept = line

    # Calculate coordinate values and cast to int - for use in plotting
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    # Draw lines with found coordinates
    img_with_overlay = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

    return img_with_overlay


def overlayROI(img, pts):
    # For visualizing ROI on original frame - not used in lane finding algorithm

    img_with_overlay = img

    img_with_overlay = cv2.line(img_with_overlay, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]),
                                (255, 0, 0), 3)                                                         # blue
    img_with_overlay = cv2.line(img_with_overlay, (pts[1][0], pts[1][1]), (pts[2][0], pts[2][1]),
                                (255, 255, 0), 3)                                                       # cyan
    img_with_overlay = cv2.line(img_with_overlay, (pts[2][0], pts[2][1]), (pts[3][0], pts[3][1]),
                                (0, 255, 0), 4)                                                         # green
    img_with_overlay = cv2.line(img_with_overlay, (pts[3][0], pts[3][1]), (pts[0][0], pts[0][1]),
                                (0, 0, 255), 5)                                                         # red
    return img_with_overlay


def ShowImage(title, image):

    # Function: Decrease number of lines needed to display an image

    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return
