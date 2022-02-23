import time

import cv2
from hand_segmentation import segment_main
from recog_hand_sign import recognise
from StackFrame import stackImages

# setting up webcam
cap = cv2.VideoCapture(0)
# webcam output frame config
cap.set(3, 640)  # width of frames
cap.set(4, 480)  # height of frames
cap.set(10, 100)  # brightness of frames
top, right, bottom, left = 10, 390, 225, 630


def back():
    print("reload")


# calling and processing each frames for infinity times.
num_frames = 0
refresh = False
pTime = 0
fps = 0

while True:
    # rading current frame
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # segmenting the hand
    clone, seg_image, cnt = segment_main(frame, num_frames, top, right, bottom, left, refresh)
    if refresh:
        refresh = False
    # recognise ASL sign
    if cnt is not None:
        pred = recognise(seg_image)
    else:
        pred = "Hand not found"

    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.rectangle(clone, (left + 1, bottom), (right - 1, bottom + 50), (0, 255, 0), -1)
    cv2.putText(clone, pred, (410, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # measuring the fps
    cTime = time.time()
    if num_frames % 30 == 0:
        fps = 1 // (cTime - pTime)
        # print(fps)
    pTime = cTime

    cv2.putText(clone, f"FPS : {int(fps)}", (530, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # preparing the window frame
    imagearray = ([clone, seg_image])
    imagestack = stackImages(0.8, imagearray)

    num_frames += 1
    cv2.imshow("hand image dataset generator", imagestack)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        refresh = True
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
