import cv2
import mediapipe as mp


class HandDetector:

    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img):
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, c = img.shape
            for id, lm in enumerate(hand.landmark):
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                lmList.append((cx, cy))
        return lmList

    def fingersUp(self, lmList):
        fingers = [0, 0, 0, 0, 0]

        if len(lmList) == 0:
            return fingers

        if lmList[4][0] > lmList[3][0]:
            fingers[0] = 1

        tipIds = [8, 12, 16, 20]
        pipIds = [6, 10, 14, 18]

        for i in range(4):
            tip = tipIds[i]
            pip = pipIds[i]
            if lmList[tip][1] < lmList[pip][1]:
                fingers[i + 1] = 1

        return fingers