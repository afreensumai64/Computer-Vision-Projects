import cv2
import numpy as np
from hand_tracking import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("ERROR: Camera not opening. Try changing VideoCapture(0) to (1).")
    exit()

detector = HandDetector()


cv2.namedWindow("Air Canvas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Air Canvas", 800, 450)      
cv2.moveWindow("Air Canvas", 100, 100)        

colorNames = ["Purple", "Blue", "Green", "Eraser"]
colorList = [
    (255, 0, 255),   # Purple
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 0),       # Eraser
]
drawColor = colorList[0]

brushThickness = 15
eraserThickness = 60
maxJumpDistance = 80   # ignore sudden jumps (gesture-switch glitches)

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

xp, yp = 0, 0


def drawHeader(img):
    boxWidth = 1280 // len(colorList)
    for i in range(len(colorList)):
        color = colorList[i]
        x1 = i * boxWidth
        x2 = x1 + boxWidth
        fillColor = color if color != (0, 0, 0) else (50, 50, 50)
        cv2.rectangle(img, (x1, 0), (x2, 100), fillColor, cv2.FILLED)
        cv2.putText(img, colorNames[i], (x1 + 10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return boxWidth


while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    boxWidth = drawHeader(img)

    modeText = "No hand detected"

    if len(lmList) != 0:
        fingers = detector.fingersUp(lmList)

        x1, y1 = lmList[8]
        x2, y2 = lmList[12]

        if fingers[1] == 1 and fingers[2] == 1:
            modeText = "SELECTION MODE - move to top bar"
            xp, yp = 0, 0
            if y1 < 100:
                idx = x1 // boxWidth
                if idx < len(colorList):
                    drawColor = colorList[idx]
            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawColor, cv2.FILLED)

        elif fingers[1] == 1 and fingers[2] == 0:
            modeText = "DRAW MODE"
            cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            distance = ((x1 - xp) ** 2 + (y1 - yp) ** 2) ** 0.5

            if distance < maxJumpDistance:
                thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
            xp, yp = x1, y1
        else:
            modeText = "IDLE - fingers: " + str(fingers)
            xp, yp = 0, 0
    else:
        xp, yp = 0, 0

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # ---- Status text ----
    cv2.putText(img, modeText, (10, 690),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # ---- Current color indicator (bottom right) ----
    cv2.rectangle(img, (1180, 650), (1260, 700), drawColor, cv2.FILLED)
    cv2.rectangle(img, (1180, 650), (1260, 700), (255, 255, 255), 2)

    cv2.imshow("Air Canvas", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    # ---- Keyboard shortcuts as backup color switch ----
    if key == ord('1'):
        drawColor = colorList[0]
    if key == ord('2'):
        drawColor = colorList[1]
    if key == ord('3'):
        drawColor = colorList[2]
    if key == ord('4'):
        drawColor = colorList[3]

cap.release()
cv2.destroyAllWindows()