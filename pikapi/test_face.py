import cv2
import openvino_open_model_zoo_toolkit.open_model_zoo_toolkit as omztk


omz = omztk.openvino_omz()

facedet = omz.faceDetector()
agegen  = omz.ageGenderEstimator()
hp      = omz.headPoseEstimator()
emo     = omz.emotionEstimator()
lm      = omz.faceLandmarksEstimator()

import time

from scipy.spatial.transform.rotation import Rotation

import numpy as np


times = []
times2 = []

cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FPS, 60)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    img = frame

    start = time.time()
    detected_faces = facedet.run(img)

    for face in detected_faces:
        start2 = time.time()
        face_img = omztk.ocv_crop(img, face[3], face[4], scale=1.3)  # Crop detected face (x1.3 wider)
        landmarks         = lm.run(face_img)                         # Estimate facial landmark points
        # Example: landmarks = [(112, 218), (245, 192), (185, 281), (138, 369), (254, 343)]

        # face_lmk_img = face_img.copy()                               # Copy cropped face image to draw markers on it
        # for lmk in landmarks:
        #     cv2.drawMarker(face_lmk_img, lmk, (255,0,0), markerType=cv2.MARKER_TILTED_CROSS, thickness=4)  # Draw markers on landmarks
        # cv2.imshow('cropped face with landmarks', face_lmk_img)
        # cv2.waitKey(2 * 1000)  # 2 sec                               # Display cropped face image with landmarks

        yaw, pitch, roll = hp.run(face_img)                          # Estimate head pose (=head rotation)
        # rot = Rotation.from_euler('xyz', [yaw, pitch, roll])
        # print(rot.as_rotvec())
        # Example: yaw, pitch, roll = -2.6668947, 22.881355, -5.5514703
        face_rot_img = omztk.ocv_rotate(face_img, roll)              # Correct roll to be upright the face

        age, gender, prob = agegen.run(face_rot_img)                 # Estimate age and gender
        # print(age,gender,prob)
        # Example: age, gender, prob = 23, female, 0.8204694
        emotion           = emo.run(face_rot_img)                    # Estimate emotion
        # Example: emotion = 'smile'

        # print(age, gender, emotion, landmarks)

        # cv2.imshow('cropped and rotated face', face_rot_img)
        # cv2.waitKey(2 * 1000)  # 2 sec

        # cv2.rectangle(img, face[3], face[4], (255,0,0), 2)
        times2.append(time.time() - start2)

    times.append(time.time() - start)
    pressed_key = cv2.waitKey(1)      # 3 sec
    if pressed_key == ord("a"):
        break
    cv2.imshow('result', img)

import plotly.graph_objects as go

import numpy as np

fig = go.Figure()
fig.add_trace(go.Box(y=times, name="Detection/Emotion/Gender/Pose"))
fig.add_trace(go.Box(y=times2, name="Emotion/Gender/Pose"))

fig.show()