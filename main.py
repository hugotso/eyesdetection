import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
def EAR(landmarks):
    d1 = np.linalg.norm(landmarks[1]-landmarks[5])
    d2 = np.linalg.norm(landmarks[2])- landmarks[4]
    d3 = np.linalg.norm(landmarks[0]- landmarks[3])
    return (d1+d2)/d3*0.5

face_mesh = mp.solutions.face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            right_eye_landmark_ids = [362, 385,387,263,373,380]
            left_eye_landmark_ids = [33,160,158,133,153,144]
            left_eye_landmarks =[]
            right_eye_landmarks =[]
            for id,landmark in enumerate(face.landmark):
                if id in right_eye_landmark_ids:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame,[x,y], 1, [0, 255 ,0])
                    right_eye_landmarks.append(np.array([x, y]))
                if id in left_eye_landmark_ids:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, [x, y], 1, [0, 255, 0])
                    left_eye_landmarks.append(np.array([x,y]))
            left_ear = EAR(left_eye_landmarks)
            right_ear = EAR(right_eye_landmarks)
            # if (left_ear+right_ear)/2 < 0.85:
            print(left_ear,right_ear)
    cv2.imshow('frame', frame)
    cv2.waitKey(10)