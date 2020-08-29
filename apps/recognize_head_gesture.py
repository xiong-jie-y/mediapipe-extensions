# This is an example to run the graph in python.
import datetime
from multiprocessing import process
import time
# import IPython

import numpy as np
from pikapi.head_gestures import YesOrNoEstimator
import pikapi.logging
from collections import defaultdict
import cv2

from pikapi.graph_runner_cpu import GraphRunnerCpu

import click

@click.command()
@click.option('--camera-id', default=0, type=int)
@click.option('--running_mode', default="gpu")
@click.option('--only-save-detections', is_flag=True)
@click.option('--log-tag', default="default_tag")
def recognize_head_gesture_live(camera_id, running_mode, only_save_detections, log_tag):
    cap = cv2.VideoCapture(camera_id)

    if running_mode == "cpu":
        print("Running on cpu mode")
        runner = GraphRunnerCpu(
            "graphs/face_mesh_desktop_live_any_model_cpu.pbtxt", 
            ["multi_face_landmarks"], 
            {
                # "detection_model_file_path": "modules/face/face_detection_front_128_float16_quant.tflite",
                "detection_model_file_path": "modules/face/face_detection_front_128_full_integer_quant.tflite",
                # "detection_model_file_path": "modules/face/face_detection_front_128_integer_quant.tflite",
                # "detection_model_file_path": "mediapipe/models/face_detection_front.tflite",
                # "landmark_model_file_path": "mediapipe/modules/face_landmark/face_landmark.tflite",
                # "landmark_model_file_path": "modules/face/face_landmark_192_float16_quant.tflite",
                "landmark_model_file_path": "modules/face/face_landmark_192_full_integer_quant.tflite",
                # "landmark_model_file_path": "modules/face/face_landmark_192_integer_quant.tflite",
            }
        )
    else:
        # Importing here to avoid compile this module, when using cpu.
        from pikapi import graph_runner
        runner = graph_runner.GraphRunner(
            "mediapipe/graphs/face_mesh/face_mesh_desktop_live_gpu.pbtxt", [
                "multi_face_landmarks"]
        )

    log = pikapi.logging.HumanReadableLog()

    numpy_array_lists = defaultdict(list)



    # import head_gestures
    yes_or_no_estimator = YesOrNoEstimator()

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
        orgHeight, orgWidth = gray.shape[:2]
        size = (int(orgWidth/2), int(orgHeight/2))
        processed_frame = cv2.resize(gray, size)

        
        processed_frame = runner.process_frame(gray)

        if running_mode == "gpu":
            processed_frame = processed_frame.reshape((480, 640, 4))
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        # log.add_data("multi_hand_landmarks", current_time, runner.get_normalized_landmark_lists("multi_hand_landmarks"))
        multi_face_landmarks = runner.get_normalized_landmark_lists(
            "multi_face_landmarks")

        blank_image = np.zeros((height,width,3), np.uint8)
        if multi_face_landmarks is not None:
            if running_mode == "cpu":
                multi_face_landmarks = [np.array([
                    [point.x, point.y, point.z]
                    for point in multi_face_landmark.landmark])
                    for multi_face_landmark in multi_face_landmarks]

            state = yes_or_no_estimator.get_state(
                pikapi.logging.TimestampedData(current_time, multi_face_landmarks))

            if state == 1:
                cv2.putText(blank_image, "Yes", (0, 50), cv2.FONT_HERSHEY_PLAIN, 4.0,
                    (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(processed_frame, "Yes", (0, 50), cv2.FONT_HERSHEY_PLAIN, 4.0,
                                    (0,0,0), 1, cv2.LINE_AA)
                # print("yes")
                
            elif state == -1:
                cv2.putText(blank_image, "No", (0, 50), cv2.FONT_HERSHEY_PLAIN, 4.0,
                    (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(processed_frame, "No", (0, 50), cv2.FONT_HERSHEY_PLAIN, 4.0,
                    (0, 0, 0), 1, cv2.LINE_AA)
                # print("no")

            for face_landmark in multi_face_landmarks:
                for point in face_landmark:
                    # print((int(point[0] * width), int(point[1] * height)))
                    cv2.circle(blank_image, (int(point[0] * width), int(point[1] * height)), 3, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

        # cv2.imshow("Frame", blank_image)
        cv2.imshow("Frame", processed_frame)

        log.add_data("multi_face_landmarks", current_time,
                    multi_face_landmarks)
        if only_save_detections:
            log.add_image("input_frame", current_time, blank_image)
        else:
            log.add_image("input_frame", current_time, processed_frame)


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
    log.save(f"output-data/{log_tag}_{datetime_now}",
            video_option={"frame_rate": cap.get(cv2.CAP_PROP_FPS)})

def main():
    recognize_head_gesture_live()

if __name__ == "__main__":
    main()