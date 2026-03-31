import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, static_image_mode=False,
                 max_num_faces=1,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.drawSpec = self.mpDraw.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 255, 0))

    def findFaceMech(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_FACE_OVAL, landmark_drawing_spec=self.drawSpec,
                                               connection_drawing_spec=self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # print(id, x, y)
                    cv2.putText(img, str(id), (x, y),
                                cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
                    face.append([id, x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture("videos/video7.mp4")
    pTime = 0
    detector = FaceMeshDetector(max_num_faces=2)

    while True:
        success, img = cap.read()

        img, faces = detector.findFaceMech(img)

        if len(faces) != 0:
            # print(len(faces))
            # print(faces[0])
            cv2.circle(img, (faces[0][1][1], faces[0][1]
                       [2]), 7, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow('Video', img)
        cv2.waitKey(10)


if __name__ == '__main__':
    main()
