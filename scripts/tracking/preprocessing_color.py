"""
    This script can be used to pick the size of the blurring kernel and the appropriate contour area range using a color image.
    To start, please specify the color range and ROI that you identified in the previous script.
    Slide the trackbar until you find the ideal value for each parameter. 
    Ideally, you should pick up all relevant contours (i.e. the blocks), while ignoring any noise.
"""

import argparse

import cv2
import numpy as np
from scripts.tracking.utils import collect_as


def pick_preprocessing(ROI_X, ROI_Y):

    global preprocessing_params, img

    ROI_Y_min, ROI_Y_max = ROI_Y
    ROI_X_min, ROI_X_max = ROI_X

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    img = img[ROI_Y_min:ROI_Y_max, ROI_X_min:ROI_X_max]

    cv2.namedWindow("Trackbar")
    cv2.createTrackbar("blur_size", "Trackbar", 0, 20, blur_change)
    cv2.createTrackbar("area_min", "Trackbar", 0, 100, area_min_change)
    cv2.createTrackbar("area_max", "Trackbar", 0, 100, area_max_change)
    preprocessing()

    while True:
        if cv2.waitKey(500) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit()


def blur_change(new_val):
    change_params("blur_size", 2 * new_val + 1)


def area_min_change(new_val):
    change_params("area_min", 10 * new_val)


def area_max_change(new_val):
    change_params("area_max", 100 * new_val)


def change_params(name, value):
    global preprocessing_params
    preprocessing_params[name] = value
    print(
        "Blurring =",
        preprocessing_params["blur_size"],
        "| Minimum Area  =",
        preprocessing_params["area_min"],
        "| Maximum Area = ",
        preprocessing_params["area_max"],
    )
    preprocessing()


def preprocessing():

    median = cv2.medianBlur(img, preprocessing_params["blur_size"])
    thresh = cv2.inRange(median, l, u)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    copy = cv2.cvtColor(np.copy(img), cv2.COLOR_HSV2RGB)

    for c in cnts:
        a = cv2.contourArea(c)
        if (
            a > preprocessing_params["area_min"]
            and a < preprocessing_params["area_max"]
        ):
            cv2.drawContours(copy, [c], 0, (255, 255, 255), 2)

    width = img.shape[1] // 2
    height = img.shape[0] // 2
    dim = (width, height)

    # resize image
    resized = cv2.resize(copy, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("", resized)


if __name__ == "__main__":

    preprocessing_params = {"blur_size": 7, "area_min": 0, "area_max": 0}

    ap = argparse.ArgumentParser()

    ap.add_argument("-v", "--video", default="", help="Video to use for preprocessing")

    ap.add_argument(
        "-y",
        "--ROI_Y",
        help="Define vertical ROI boundaries",
        type=int,
        nargs="+",
        required=True,
        action=collect_as(tuple),
    )

    ap.add_argument(
        "-x",
        "--ROI_X",
        help="Define horizontal ROI boundaries",
        type=int,
        nargs="+",
        required=True,
        action=collect_as(tuple),
    )

    ap.add_argument(
        "-l",
        "--lower_colorspace",
        help="Define lower colorspace",
        type=int,
        nargs="+",
        required=True,
        action=collect_as(np.array),
    )

    ap.add_argument(
        "-u",
        "--upper_colorspace",
        help="Define upper colorspace",
        type=int,
        nargs="+",
        required=True,
        action=collect_as(np.array),
    )

    args = ap.parse_args()

    vidcap = cv2.VideoCapture(args.video)
    _, img = vidcap.read()

    l, u = args.lower_colorspace, args.upper_colorspace

    pick_preprocessing(args.ROI_X, args.ROI_Y)
