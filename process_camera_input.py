# This is an example to run the graph in python.
import datetime
import time
import result
from collections import defaultdict
import cv2

import graph_runner

cap = cv2.VideoCapture(4)

# runner = graph_runner.GraphRunner(
#         "mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt",
#         ["channel2", "channel3"]
#         )

# runner = graph_runner.GraphRunner(
#         "graphs/hand_face_detection_no_gating.pbtxt", ["multi_hand_landmarks", "multi_face_landmarks"])

# runner = graph_runner.GraphRunner(
#     "mediapipe/graphs/pose_tracking/upper_body_pose_tracking_gpu.pbtxt", []
# )

runner = graph_runner.GraphRunner(
    "mediapipe/graphs/face_mesh/face_mesh_desktop_live_gpu.pbtxt", [
        "multi_face_landmarks"]
)

tag_name = "face_ui"


log = result.HumanReadableLog()

numpy_array_lists = defaultdict(list)

while(True):
    current_time = time.time()
    ret, frame = cap.read()
    if frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = runner.process_frame(gray)
    processed_frame = processed_frame.reshape((480, 640, 4))
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Frame", processed_frame)

    # log.add_data("multi_hand_landmarks", current_time, runner.get_normalized_landmark_lists("multi_hand_landmarks"))
    multi_face_landmarks = runner.get_normalized_landmark_lists(
        "multi_face_landmarks")
    log.add_data("multi_face_landmarks", current_time,
                 multi_face_landmarks)
    log.add_image("input_frame", current_time, processed_frame)

    if cv2.waitKey(10) > 0:
        break

datetime_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log.save(f"data/{tag_name}_{datetime_now}",
         video_option={"frame_rate": cap.get(cv2.CAP_PROP_FPS)})
