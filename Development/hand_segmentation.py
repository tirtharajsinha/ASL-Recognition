# organize imports
import cv2
import imutils
import numpy as np

# global variables
bg = None


# --------------------------------------------------
# To find the running average over the background
# --------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


# ---------------------------------------------
# To segment the region of hand in the image
# ---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


# -----------------
# MAIN FUNCTION
# -----------------
def segment_main(frame, num_frames, top, right, bottom, left, refresh):
    # initialize weight for running average

    aWeight = 0.5

    # region of interest (ROI) coordinates

    # clone the frame
    clone = frame.copy()
    blank = np.zeros((512, 512, 1), dtype="uint8")
    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # get the ROI
    roi = frame[top:bottom, right:left]

    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # to get the background, keep looking till a threshold is reached
    # so that our running average model gets calibrated

    if num_frames < 30 or refresh:
        run_avg(gray, aWeight)
        return clone, blank, None
    # if num_frames > 31 and num_frames % 500 == 0:
    #     run_avg(gray, aWeight)
    #     return clone, gray, None
    # segment the hand region
    hand = segment(gray)

    # check whether hand region is segmented
    if hand is not None:
        # if yes, unpack the thresholded image and
        # segmented region
        (thresholded, segmented) = hand

        # draw the segmented region and display the frame
        cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

        return clone, thresholded, segmented
    else:
        return clone, blank, None
