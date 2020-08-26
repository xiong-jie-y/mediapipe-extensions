import time

import streamlit as st


import graph_runner
import cv2

@st.cache(allow_output_mutation=True)
def get_runner():
    return graph_runner.GraphRunner(
        "mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt")
runner = get_runner()

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

# [frame_idx:frame_idx+10]

times = []
output_frames = []
for frame in frames:
    start = time.time()
    output_frames.append(runner.process_frame(frame))
    end = time.time()
    times.append(end - start)

import plotly.graph_objects as go

import numpy as np

import numpy as np
# st.write(1.0 / np.array(times))
st.write(1.0 / np.mean(times))

x0 = np.random.randn(2000)
x1 = np.random.randn(2000) + 1

fig = go.Figure()
fig.add_trace(go.Histogram(x=times))
# fig.add_trace(go.Histogram(x=x1))

# The two histograms are drawn on top of another
fig.update_layout(barmode='stack')
# fig.show()
st.write(fig)
# st.write(output_frames[-1])

# st.write(new_frame.shape)

new_frame = output_frames[-1]
new_frame = new_frame.reshape((480, 640, 4))
st.image(new_frame)


# cameravtuber2.GraphRunner("mediapipe/graphs/hand_tracking/hand_face_tracking_desktop.pbtxt")
