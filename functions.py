import numpy as np
import cv2

def findLaneLines(frame):

    # In the order of top left, top right, bottom right, bottom left
    roi_pts = np.array([[590, 530], [682, 530], [1100, 660], [124, 660]], np.int32)

    # In the order of top left, top right, bottom right, bottom left
    src_pts = np.float32(
        [
            [520.0, 560.0],
            [740.0, 560.0],
            [1000.0, 700.0],
            [270.0, 700.0],
        ]
    )

    # Warp lane in front of car to bird eye view
    warped_img, M, Minv = warp_image(frame, src_pts)
    img = warped_img
    ShowImage("warped", img)
    usingUnwarped = False

    # Threshold warped image (grayscale, and binary)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Canny Edge Detection
    img = cv2.Canny(img, 100, 200)
    ShowImage('canny', img)

    # Apply Gaussian Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    if usingUnwarped == True:
        # Region of Interest - For un-warped frame
        # Create mask
        mask = np.zeros_like(img)
        # Fill mask
        cv2.fillPoly(mask, [roi_pts], 255) # This is the problem....
        # Join mask and img
        masked_img = cv2.bitwise_and(img, mask)
        ShowImage('ROI', masked_img)
        img = masked_img



    # Hough Lines
    minLineLength = 1100
    maxLineGap = 10
    #lines = cv2.HoughLines(masked_img, 1, np.pi/180, 100)

    #                          edges   rho   theta   thresh  min length, max gap:
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 10,      100,         10)

    # Draw Hough Lines
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)

    # Display Warped image
    ShowImage('Frame with lines', img)

    # FIXME: Just for now so we have a return value
    lane_pts = roi_pts

    return lane_pts

def warp_image(img, src_pts):
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
    )

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_img = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)

    return warped_img, M, Minv

def overlay(img, pts):

    # FIXME: Currently Draws a rectangle (for showing the warped image) --- Needs to just be the lane lines in the end
    img_with_overlay = cv2.line(img, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), (255, 0, 0), 3)    # blue
    img_with_overlay = cv2.line(img_with_overlay, (pts[1][0], pts[1][1]), (pts[2][0], pts[2][1]),
                                (255, 255, 0), 3)                                                       # cyan
    img_with_overlay = cv2.line(img_with_overlay, (pts[2][0], pts[2][1]), (pts[3][0], pts[3][1]),
                                (0, 255, 0), 4)                                                         # green
    img_with_overlay = cv2.line(img_with_overlay, (pts[3][0], pts[3][1]), (pts[0][0], pts[0][1]),
                                (0, 0, 255), 5)                                                         # red

    return img_with_overlay

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = (255,) 
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def ShowImage(title, image):

    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return
