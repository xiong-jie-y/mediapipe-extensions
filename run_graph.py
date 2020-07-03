import time

import streamlit as st


import graph_runner
import cv2

runner = graph_runner.GraphRunner(
    "mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt")


@st.cache(allow_output_mutation=True)
def get_frames():
    cap = cv2.VideoCapture("hand_motion2.mp4")

    frames = []

    while(True):
        ret, frame = cap.read()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(gray)

    return frames


frames = get_frames()
frame_idx = st.slider('frame idx', 0, len(frames))
# st.image(frames[frame_idx])
# st.write(frames[frame_idx].shape)
frame = frames[frame_idx]
start = time.time()
new_frame = runner.process_frame(frame)
end = time.time()
st.write(end - start)
st.write(new_frame.shape)

new_frame = new_frame.reshape((480, 640, 4))
st.image(new_frame)


# cameravtuber2.GraphRunner("mediapipe/graphs/hand_tracking/hand_face_tracking_desktop.pbtxt")
