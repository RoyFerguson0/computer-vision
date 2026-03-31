import time

import cv2

from PoseModule import PoseDetector

cap = cv2.VideoCapture('videos/video1.mp4')
pTime = 0

detector = PoseDetector()

while True:
    success, img = cap.read()

    img = detector.findPose(img)
    lmList = detector.findPostion(img, draw=False)

    if len(lmList) != 0:
        print(lmList[14])

        cv2.circle(img, (lmList[14][1], lmList[14][2]),
                   15, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('Video', img)

    cv2.waitKey(1)
