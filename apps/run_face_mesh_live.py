# This is an example to run the graph in python.
import IPython
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


def get_focal_length(camera_id):
    cap = cv2.VideoCapture(camera_id)
    width = cap.get(3)
    height = cap.get(4)
    cap.release()

    return get_intrinsic(int(width), int(height)).ppx

# def face_mesh_to_camera_coordinate(face_iris_landmark, left_mm, right_mm):
#     import IPython; IPython.embed()
#     left_center_iris = face_iris_landmark[468, :]
#     right_center_iris = face_iris_landmark[468 + 5, :]


#     left_relatives = np.copy(face_iris_landmark)
#     left_relatives[:, 2] =  left_relatives[:, 2] - left_center_iris[2] + left_mm
#     left_relatives[:, 2] =  left_relatives[:, 2] - left_center_iris[2] + left_mm
    

@click.command()
@click.option('--camera-id', '-c', default=0, type=int)
@click.option('--relative-depth', is_flag=True)
def cmd(camera_id, relative_depth):
    cap = cv2.VideoCapture(camera_id)

    if not relative_depth:
        focal_length = get_focal_length(camera_id)
        runner = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/iris_tracking_gpu.pbtxt", [
                "face_landmarks_with_iris", "left_iris_depth_mm", "right_iris_depth_mm"],
            pmu.create_packet_map({"focal_length_pixel": focal_length})
        )
    else:
        # runner = pikapi.graph_runner.GraphRunner(
        #     "pikapi/graphs/face_mesh_desktop_live_any_model.pbtxt", [
        #         "multi_face_landmarks"],
        #     pmu.create_packet_map({
        #         "detection_model_file_path": pikapi.get_data_path("models/face_detection_front.tflite"),
        #         "landmark_model_file_path": pikapi.get_data_path("models/face_landmark.tflite"),
        #     })
        # )
        runner = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/multi_hand_tracking_gpu.pbtxt", [
                "multi_hand_landmarks"],
            {})

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

        multi_face_landmarks = runner.get_normalized_landmark_lists(
            "multi_hand_landmarks")

        for face_landmark in multi_face_landmarks:
            if len(face_landmark) != 0:
                # print(face_mesh_to_camera_coordinate(face_landmark, left_mm, right_mm))
                for point in face_landmark:
                    cv2.circle(blank_image, (int(point[0] * frame.shape[1]), int(
                        point[1] * frame.shape[0])), 3, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                # print(np.mean([point[2] for point in face_landmark]))

        cv2.imshow("Frame", processed_frame)

        pressed_key = cv2.waitKey(3)
        if pressed_key == ord("a"):
            break


def main():
    cmd()


if __name__ == "__main__":
    main()
