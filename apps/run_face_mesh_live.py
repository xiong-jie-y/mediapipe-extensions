# This is an example to run the graph in python.
import cv2

from pikapi import graph_runner

import click


@click.command()
@click.option('--camera-id', '-c', default=0, type=int)
def cmd(camera_id):
    cap = cv2.VideoCapture(camera_id)

    runner = graph_runner.GraphRunner(
        "mediapipe/graphs/face_mesh/face_mesh_desktop_live_gpu.pbtxt", [
            "multi_face_landmarks"]
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

        multi_face_landmarks = runner.get_normalized_landmark_lists(
            "multi_face_landmarks")

        for face_landmark in multi_face_landmarks:
            for point in face_landmark:
                cv2.circle(processed_frame, (int(point[0] * frame.shape[1]), int(
                    point[1] * frame.shape[0])), 3, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

        cv2.imshow("Frame", processed_frame)

        pressed_key = cv2.waitKey(3)
        if pressed_key == ord("a"):
            break


def main():
    cmd()


if __name__ == "__main__":
    main()
