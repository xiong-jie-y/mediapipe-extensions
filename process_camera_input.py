# This is an example to run the graph in python.
import cv2

import graph_runner

cap = cv2.VideoCapture(4)

# runner = graph_runner.GraphRunner(
#         "mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt",
#         ["channel2", "channel3"]
#         )

# runner = graph_runner.GraphRunner(
#         "graphs/hand_face_detection_no_gating.pbtxt", ["multi_hand_landmarks", "multi_face_landmarks"])

runner = graph_runner.GraphRunner(
    "mediapipe/graphs/pose_tracking/upper_body_pose_tracking_gpu.pbtxt", []
)

# runner = graph_runner.GraphRunner(
#     "mediapipe/graphs/face_mesh/face_mesh_desktop_live_gpu.pbtxt", []# ["multi_face_landmarks"]
# )


# runner = graph_runner.GraphRunner(
#     "mediapipe/graphs/face_mesh/face_mesh_desktop_mobile.pbtxt", ["multi_face_landmarks"]
# )

frames = []

frame_rate = cap.get(cv2.CAP_PROP_FPS)
print(cap.get(cv2.CAP_PROP_FPS))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
size = (640, 480) # 動画の画面サイズ

fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
writer = cv2.VideoWriter('./result/outtest.mp4', fmt, frame_rate, size) # ライター作成

landmark_lists_all = []

while(True):
    ret, frame = cap.read()
    if frame is None:
        break

    # import IPython; IPython.embed()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = runner.process_frame(gray)
    processed_frame = processed_frame.reshape((480, 640, 4))
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Frame", processed_frame)

    # landmark_lists_all.append(runner.get_normalized_landmark_lists("multi_hand_landmarks"))
    # landmark_lists_all.append(runner.get_normalized_landmark_lists("multi_face_landmarks"))
    writer.write(processed_frame) # 画像を1フレーム分として書き込み

    if cv2.waitKey(10) > 0:
        break

writer.release()
import numpy as np
np.savez_compressed("./result/outtest.nz", landmark_lists_all)