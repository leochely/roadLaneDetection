import numpy as np
import math
import cv2

def findLaneLines(frame):

    # In the order of top left, top right, bottom right, bottom left
    src_pts = np.float32(
        [
            [520.0, 560.0],
            [740.0, 560.0],
            [1000.0, 700.0],
            [270.0, 700.0],
        ]
    )

    usingWarped = True

    if usingWarped:
        # Warp lane in front of car to bird eye view
        warped_img, M, Minv = warp_image(frame, src_pts)
        img = warped_img
        #ShowImage("Warped Image", warped_img)

        # ROI in Warped Image (works great for the whole video)
        roi_pts = np.array([[200, 0], [1000, 0], [1000, 720], [200, 720]], np.int32)

    else:
        img = frame

        # In the order of top left, top right, bottom right, bottom left
        roi_pts = np.array([[590, 530], [682, 530], [1100, 660], [124, 660]], np.int32)


    # Threshold warped image (grayscale, and binary)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Canny Edge Detection
    img = cv2.Canny(img, 100, 200)
    #ShowImage('Canny Edge Image', img)

    # Apply Gaussian Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Create mask for ROI
    mask = np.zeros_like(img)
    # Fill mask
    cv2.fillPoly(mask, [roi_pts], 255)
    # Join mask and img
    masked_img = cv2.bitwise_and(img, mask)
    #ShowImage('Masked Image (ROI)', masked_img)
    img = masked_img

    # Hough Lines         edges, rho   theta   thresh              min length, max gap
    #lines = cv2.HoughLinesP(img, 1, np.pi/180, 10, np.array([]),      100,         200)
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, np.array([]), 100, 250)

    # Draw Hough Lines
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 1)
    #ShowImage("Hough Lines", img)

    # Convert Hough lines for a single Left and Right Line
    left_line, right_line, min_x = houghLinestoLaneLines(img, lines)

    # For Testing, Overlay on Warped Img
    overlayed_img = overlay(img, left_line)
    overlayed_img = overlay(img, right_line)
    #ShowImage("Overlayed on Warped", overlayed_img)

    # Unwarp (inverse)



    return  overlayed_img, left_line, right_line


def houghLinestoLaneLines(img, lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)
    left_line_x = []
    left_line_y = []
    min_x = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Check for infinite demoninator (Veticle line)
            if (x2 - x1) == 0:
                continue

            # Calculate Slope
            slope = (y2 - y1) / (x2 - x1)

            # Only keep extreeme values
            if math.fabs(slope) < 1:
                continue

            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if slope < 0:  # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))

                if min_x == []:
                    min_x = x1
                else:
                    if x1 < min_x:
                        min_x = x1
                    if x2 < min_x:
                        min_x = x2

                #left_line_x.extend([x1, x2])
                #left_line_y.extend([y1, y2])
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_line = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_line = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    '''
    y1 = img.shape[0]
    y2 = 0

    for line in left_lines:
        for slope, intercept in line:
            left_line_x.extend([int((y1 - intercept) / slope)])
            left_line_x.extend([int((y2 - intercept) / slope)])
    
    
    img_with_overlay = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)
    
    

    print("left: ", left_line)
    print("right: ", right_line)

    
    y1 = img.shape[0]
    y2 = 0

    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=2
    ))

    x1 = int(poly_left(y1))
    x2 = int(poly_left(y2))

    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)
    ShowImage("here", img)
    '''

    return left_line, right_line, min_x


def findSingleLines(img, lines):

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Check for infinite demoninator
            if (x2 - x1) == 0:
                continue
            # Calculate Slope
            slope = (y2 - y1) / (x2 - x1)

            # Only keep extreeme values
            if math.fabs(slope) < 1:
                continue

            # Left line for negative slope
            if slope <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            # Right line for positive slope
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])


    min_y = 0
    max_y = img.shape[0]

    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    print("Ls: ", left_x_start)
    print("Le: ", left_x_end)
    print("Rs: ", right_x_start)
    print("Re: ", right_x_end)

    cv2.line(img, (left_x_start, max_y), (left_x_end, min_y), (255, 255, 255), 5)
    cv2.line(img, (right_x_start, max_y), (right_x_end, min_y), (255, 255, 255), 5)

    return img


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

def overlay(img, line, min_x = None):

    if line is None:
        return img

    y1 = img.shape[0]
    y2 = 0

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    if min_x == None:

        img_with_overlay = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)

    else:
        offset = x1 - min_x
        x1 = x1 - offset
        x2 = x2 - offset

        img_with_overlay = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)

    return img_with_overlay


def overlayROI(img, pts):
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

    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return
