"""
    To pick a colorspace that works for you, click on the objects you want to track in 'hsv'. 
    If you are satisfied with the result, continue. Otherwise, click again or adjust the ranges of your colorspace.
"""

import cv2
import numpy as np
import argparse


image_hsv = None
pixel = (0, 0, 0)  # some default

# mouse callback function


def click_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y, x]

        # you might want to adjust the ranges(+-10, etc):
        upper = np.array([pixel[0] + 50, pixel[1] + 50, pixel[2] + 80])
        lower = np.array([pixel[0] - 50, pixel[1] - 50, pixel[2] - 80])

        print("Lower colorspace: {} | Upper colorspace: {}".format(lower, upper))

        image_mask = cv2.inRange(image_hsv, lower, upper)
        cv2.imshow("mask", image_mask)


def pick_color(image_src):

    global pixel  # so we can use it in mouse callback
    global image_hsv

    width = image_src.shape[1] // 2
    height = image_src.shape[0] // 2
    dim = (width, height)

    # resize image
    image_src = cv2.resize(image_src, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("rgb", image_src)

    cv2.namedWindow("hsv")
    cv2.setMouseCallback("hsv", click_color)

    # now click into the hsv img and look at values:
    image_hsv = cv2.cvtColor(image_src, cv2.COLOR_RGB2HSV)
    cv2.imshow("hsv", image_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default="", help="Video to use for preprocessing")
    args = ap.parse_args()

    vidcap = cv2.VideoCapture(args.video)
    _, image_src = vidcap.read()

    # select ROI function
    showCrosshair = False
    fromCenter = False

    cv2.namedWindow("First frame", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("First frame", image_src, showCrosshair, fromCenter)

    # print rectangle points of selected ROI
    print("ROI_Y = [", int(roi[1]), int(roi[1] + roi[3]), "] | ROI_X = [", int(roi[0]), int(roi[0] + roi[2]), "]")

    # Crop selected ROI from raw image
    image_src = image_src[
        int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])
    ]

    pick_color(image_src)
