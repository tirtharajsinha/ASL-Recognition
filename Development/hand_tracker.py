import sys

import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(self, staticImageMode=False, maxNumHands=2, minDetectionConfidence=0.5, trackCon=0.5):
        self.results = None
        self.staticImageMode = staticImageMode
        self.maxNumberHands = maxNumHands
        self.minDetectionConfidence = minDetectionConfidence
        self.trackCon = trackCon
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.staticImageMode,
            max_num_hands=self.maxNumberHands,
            min_detection_confidence=self.minDetectionConfidence,
            min_tracking_confidence=self.trackCon)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        # for rect in self.results.hand_rects:
        #     print(rect)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img, handLms,
                                                   self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, maxHandNo=2, draw=True):
        mainlmlist = []
        handsType = []
        # handtype=[0,0]
        if self.results.multi_handedness:
            for hand in self.results.multi_handedness:
                # print(hand)
                # print(hand.classification)
                # print(hand.classification[0])
                handType = hand.classification[0].label
                print(handType)
                handsType.append(handType)

        # print(len(self.results.multi_hand_landmarks[0]))
        if self.results.multi_hand_landmarks:
            for myHand in self.results.multi_hand_landmarks:
                lmList = []
                if self.results.multi_hand_landmarks:

                    for pid, lm in enumerate(myHand.landmark):
                        # print(id, lm)
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # print(id, cx, cy)
                        lmList.append([pid, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                mainlmlist.append(lmList)

        return mainlmlist, handsType

    def getBoundedBox(self, lmList):
        top, right, bottom, left = sys.maxsize, 0, 0, sys.maxsize
        for pts in lmList:
            if pts[1] < left:
                left = pts[1]
            if pts[1] > right:
                right = pts[1]

            if pts[2] > bottom:
                bottom = pts[2]
            if pts[2] < top:
                top = pts[2]

        return top, right, bottom, left
