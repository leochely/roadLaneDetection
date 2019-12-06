import numpy as np
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

    # Warp lane in front of car to bird eye view
    warped_img, M, Minv = warp_image(frame, src_pts)

    # Display Warped image
    ShowImage('warped_img', warped_img)

    # Threshold warped image



    # FIXME: Just for now so we have a return value
    lane_pts = src_pts

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


def ShowImage(title, image):

    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return
