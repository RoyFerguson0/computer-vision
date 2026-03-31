import cv2
import numpy as np
import time

from PoseModule import PoseDetector

# cap = cv2.VideoCapture("videos/video8.mp4")
cap = cv2.VideoCapture(0)
# img = cv2.imread('images/test.jpg')
detector = PoseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, draw=False)
    lmList = detector.findPostion(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        # # Right Arm
        # angle = detector.findAngle(img, 12, 14, 16)

        # Left Arm
        angle = detector.findAngle(img, 11, 13, 15)

        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (650, 100))
        # print(angle, per)

        # check for the bicep curls
        colour = (255, 0, 255)
        if per == 100:
            colour = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            colour = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        # print(count)

        # Draw Bar
        cv2.rectangle(img, (1100, 100), (1175, 650), colour, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), colour, cv2.FILLED)
        cv2.putText(img, f"{int(per)}%", (1100, 75),
                    cv2.FONT_HERSHEY_PLAIN, 4, colour, 4)

        # Draw Curl count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"{str(int(count))}", (45, 670),
                    cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 35)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (50, 100),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)
    cv2.imshow("Video", img)
    cv2.waitKey(1)
