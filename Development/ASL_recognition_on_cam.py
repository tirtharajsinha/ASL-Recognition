import time

import cv2
import numpy as np

from hand_segmentation import segment_main
from recog_hand_sign import recognise
from StackFrame import stackImages
from hand_tracker import handDetector

# version v1.5

# setting up webcam
cap = cv2.VideoCapture(0)
# webcam output frame config
cap.set(3, 640)  # width of frames
cap.set(4, 480)  # height of frames
cap.set(10, 100)  # brightness of frames

handDetectorModel = handDetector(minDetectionConfidence=0.50)

# calling and processing each frames for infinity times.
num_frames = 0
refresh = False
pTime = 0
while True:
    # rading current frame
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # top, right, bottom, left = 480, 0, 0, 640

    # calling handDetectionModel
    clone = handDetectorModel.findHands(frame.copy())

    mainlmList, handsType = handDetectorModel.findPosition(clone, draw=False)
    for pid, lmList in enumerate(mainlmList):
        top, right, bottom, left = handDetectorModel.getBoundedBox(lmList)
        if len(lmList) > 0:
            pred = handsType[pid]+" hand"
            cv2.rectangle(clone, (left - 20, top - 20), (right + 20, bottom + 20), (0, 255, 0), 2)
            # cv2.rectangle(clone, (left + 1, bottom), (right - 1, bottom + 50), (0, 255, 0), -1)
            cv2.putText(clone, pred, (left + 20, bottom + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (82, 82, 255), 2)

    # segmenting the hand
    seg_image = np.zeros((512, 512, 1), dtype="uint8")

    # measuring the fps
    cTime = time.time()
    fps = 1 // (cTime - pTime)
    pTime = cTime

    cv2.putText(clone, f"FPS : {int(fps)}", (530, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    num_frames += 1
    # preparing the window frame
    # imagearray = ([clone, seg_image])
    # imagestack = stackImages(0.8, imagearray)

    cv2.imshow("ASL Recognition", clone)

    if cv2.waitKey(1) & 0xFF == ord('r'):
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
