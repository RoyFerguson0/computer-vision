import time

import cv2

from HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector()
pTime = 0
while True:
    success, img = cap.read()

    detector.findHand(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList[0])
        cv2.circle(img, (lmList[0][1], lmList[0][2]),
                   15, (255, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('Video', img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
