# This is an example to run the graph in python.
import datetime
import time

import IPython
import numpy as np
from pika.head_gestures import YesOrNoEstimator
import pika.logging
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


log = pika.logging.HumanReadableLog()

numpy_array_lists = defaultdict(list)



# import head_gestures
yes_or_no_estimator = YesOrNoEstimator()

import pika.logging

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
labels = []

import apps.controll_window
from multiprocessing import Process

from multiprocessing import Value

label_state = Value('i', 0)
action = Value('i', 0)
p = Process(target=apps.controll_window.create_window, args=(label_state, action))
p.start()

print("start")
value_to_label = {}
value_to_label[1] = "Nodding"
value_to_label[-1] = "Shaking"
value_to_label[0] = None

while(True):
    current_time = time.time()
    ret, frame = cap.read()
    if frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = runner.process_frame(gray)
    processed_frame = processed_frame.reshape((480, 640, 4))
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

    # log.add_data("multi_hand_landmarks", current_time, runner.get_normalized_landmark_lists("multi_hand_landmarks"))
    multi_face_landmarks = runner.get_normalized_landmark_lists(
        "multi_face_landmarks")
        
    state = yes_or_no_estimator.get_state(
        pika.logging.TimestampedData(current_time, multi_face_landmarks))

    blank_image = np.zeros((height,width,3), np.uint8)
    if state == 1:
        cv2.putText(blank_image, "Yes", (0, 50), cv2.FONT_HERSHEY_PLAIN, 4.0,
            (255, 255, 255), 1, cv2.LINE_AA)
        # print("yes")
        
    elif state == -1:
        cv2.putText(blank_image, "No", (0, 50), cv2.FONT_HERSHEY_PLAIN, 4.0,
            (255, 255, 255), 1, cv2.LINE_AA)
        # print("no")

    for face_landmark in multi_face_landmarks:
        for point in face_landmark:
            # print((int(point[0] * width), int(point[1] * height)))
            cv2.circle(blank_image, (int(point[0] * width), int(point[1] * height)), 3, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

    # cv2.imshow("Frame", blank_image)
    cv2.imshow("Frame", blank_image)

    log.add_data("multi_face_landmarks", current_time,
                 multi_face_landmarks)
    log.add_image("input_frame", current_time, blank_image)

    # time.sleep(0.001)
    # if cv2.waitKey(1) > 0:
    #     break

    pressed_key = cv2.waitKey(1)
    # if pressed_key == ord("a"):
    #     log.add_data("motion_label", current_time, "Nodding")
    # elif pressed_key == ord("b"):
    #     log.add_data("motion_label", current_time, "Shaking")
    # else:
    #     log.add_data("motion_label", current_time, None)
    log.add_data("motion_label", current_time, value_to_label[label_state.value])
    if action.value:
        print("e")
        break

p.kill()

datetime_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log.save(f"data/{tag_name}_{datetime_now}",
         video_option={"frame_rate": cap.get(cv2.CAP_PROP_FPS)})
