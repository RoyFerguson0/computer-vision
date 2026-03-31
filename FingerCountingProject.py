import os

import cv2
import time

from HandTrackingModule import HandDetector

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "images/fingers"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
pTime = 0

myList.sort(key=lambda x: int(x.split('.')[0]))
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    image = cv2.resize(image, (200, 200))
    # print(f"{folderPath}/{imPath}")
    overlayList.append(image)


# print(len(overlayList))

detector = HandDetector()

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    img = detector.findHand(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []

        # Determine handedness
        handedness = "Right" if lmList[0][1] < lmList[12][1] else "Left"

        # Thumb
        if handedness == "Right":
            if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                # print("index finger open")
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
                # print("index finger open")
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                # print("index finger open")
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)

        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
        cv2.putText(img, handedness, (500, 120),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (400, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv2.imshow('Video', img)
    cv2.waitKey(1)
