# This is an example to run the graph in python.
import cv2
import numpy as np

from pikapi import graph_runner

import click

import pikapi
import pikapi.mediapipe_util as pmu

import pyrealsense2 as rs

def get_intrinsic(width, height):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    # pipeline.stop()
    cfg = pipeline.start(config)
    # Fetch stream profile for depth stream
    video_profile = cfg.get_stream(rs.stream.color)
    intri = video_profile.as_video_stream_profile().get_intrinsics()
    pipeline.stop()
    return intri


@click.command()
@click.option('--camera-id', '-c', default=0, type=int)
@click.option('--relative-depth', is_flag=True)
def cmd(camera_id, relative_depth):
    # focal_length = get_intrinsic(int(640), int(480)).ppx
    # import IPython; IPython.embed()

    cap = cv2.VideoCapture(camera_id)

    # width = cap.get(3)
    # height = cap.get(4)
    
    if not relative_depth:
        runner = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/iris_tracking_gpu.pbtxt", [
                "face_landmarks_with_iris", "left_iris_depth_mm", "right_iris_depth_mm"], 
                    pmu.create_packet_map({"focal_length_pixel": 327.3898010253906})
                )
    else:
        runner = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/face_mesh_desktop_live_any_model.pbtxt", [
                "multi_face_landmarks"], 
                pmu.create_packet_map({
                "detection_model_file_path": pikapi.get_data_path("models/face_detection_front.tflite"),
                "landmark_model_file_path": pikapi.get_data_path("models/face_landmark.tflite"),
            })
        )

    while(True):
        ret, frame = cap.read()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = runner.process_frame(gray)
        processed_frame = processed_frame.reshape(
            (frame.shape[0], frame.shape[1], 4))
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        # multi_face_landmarks = runner.get_normalized_landmark_lists(
        #     "multi_face_landmarks")

        print(processed_frame.shape)
        blank_image = np.zeros(processed_frame.shape, np.uint8)

        multi_face_landmarks = [runner.get_normalized_landmark_list(
            "face_landmarks_with_iris")]
        left_mm = runner.get_float("left_iris_depth_mm")
        right_mm = runner.get_float("right_iris_depth_mm")
        if left_mm is not None and right_mm is not None:
            # cv2.putText(blank_image, str(int((left_mm + right_mm)/2)) + "[mm]", (0, 50), cv2.FONT_HERSHEY_PLAIN, 4.0,
            #     (255, 255, 255), 1, cv2.LINE_AA)
           cv2.putText(blank_image, f"Left: {left_mm} [mm]", (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0,
                (255, 255, 255), 1, cv2.LINE_AA)
           cv2.putText(blank_image, f"Right: {right_mm} [mm]", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2.0,
                (255, 255, 255), 1, cv2.LINE_AA)


        for face_landmark in multi_face_landmarks:
            for point in face_landmark:
                cv2.circle(blank_image, (int(point[0] * frame.shape[1]), int(
                    point[1] * frame.shape[0])), 3, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            # print(np.mean([point[2] for point in face_landmark]))

        cv2.imshow("Frame", blank_image)

        pressed_key = cv2.waitKey(3)
        if pressed_key == ord("a"):
            break


def main():
    cmd()


if __name__ == "__main__":
    main()
