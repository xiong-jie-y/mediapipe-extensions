"""This is the perception server that accept RGBD camera input 
and output data through ipc (currently zeromq) in the proto message format..
"""
from numpy.core.fromnumeric import mean
from pikapi.utils.unity import realsense_vec_to_unity_char_vec
import time

from scipy.spatial.transform.rotation import Rotation
import pikapi.protos.perception_state_pb2 as ps
from pikapi.head_gestures import YesOrNoEstimator
import logging
from typing import Dict, List
import IPython
import cv2
import numpy as np

from pikapi import graph_runner

import click

import pikapi
import pikapi.mediapipe_util as pmu

import pyrealsense2 as rs
from pikapi.utils.realsense import *
from pikapi.core.camera import IMUInfo
from pikapi.recognizers.geometry.face import FaceGeometryRecognizer
from pikapi.recognizers.geometry.hand import HandGestureRecognizer
from pikapi.recognizers.geometry.body import BodyGeometryRecognizer

@click.command()
@click.option('--camera-id', '-c', default=0, type=int)
@click.option('--relative-depth', is_flag=True)
@click.option('--use-realsense', is_flag=True)
@click.option('--demo-mode', is_flag=True)
def cmd(camera_id, relative_depth, use_realsense, demo_mode):
    intrinsic_matrix = get_intrinsic_matrix(camera_id)

    face_recognizer = FaceGeometryRecognizer(intrinsic_matrix)
    hand_gesture_recognizer = HandGestureRecognizer(intrinsic_matrix)
    body_recognizer = BodyGeometryRecognizer(intrinsic_matrix)
    import zmq
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://127.0.0.1:9998")

    # Alignオブジェクト生成
    config = rs.config()
    #config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    #config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    #
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 60)
    #
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 60)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)

    from pikapi.orientation_estimator import RotationEstimator

    rotation_estimator = RotationEstimator(0.98, True)

    last_run = time.time()
    from collections import deque
    acc_vectors = deque([])
    try:
        while(True):
            last_run = time.time()
            current_time = time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            # import IPython; IPython.embed()

            # Camera pose estimation.
            acc = frames[2].as_motion_frame().get_motion_data()
            gyro = frames[3].as_motion_frame().get_motion_data()
            timestamp = frames[3].as_motion_frame().get_timestamp()
            rotation_estimator.process_gyro(
                np.array([gyro.x, gyro.y, gyro.z]), timestamp)
            rotation_estimator.process_accel(np.array([acc.x, acc.y, acc.z]))
            theta = rotation_estimator.get_theta()
            acc_vectors.append(np.array([acc.x, acc.y, acc.z]))
            if len(acc_vectors) > 200:
                acc_vectors.popleft()
            acc = np.mean(acc_vectors, axis=0)
            imu_info = IMUInfo(acc)

            # print(acc)
            # print(np.array([acc.x, acc.y, acc.z]))

            # print(theta)

            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # ret, frame = cap.read()
            # if frame is None:
            #     break

            # import IPython; IPython.embed()
            # frame[depth_image > 2500] = 0
            # frame[depth_image == 0] = 0
            # depth_image[depth_image > 2500] = 0
            # depth_image[depth_image == 0] = 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            width = frame.shape[1]
            height = frame.shape[0]

            # Prepare depth image for dispaly.
            depth_image_cp = np.copy(depth_image)
            depth_image_cp[depth_image_cp > 2500] = 0
            depth_image_cp[depth_image_cp == 0] = 0
            depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image_cp, alpha=0.08), cv2.COLORMAP_JET)
            # depth_colored = cv2.convertScaleAbs(depth_image, alpha=0.08)

            if demo_mode:
                target_image = depth_colored
            else:
                target_image = frame

            face_state = face_recognizer.get_face_state(
                gray, depth_image, target_image, imu_info)

            hand_states = \
                hand_gesture_recognizer.get_hand_states(
                    gray, depth_image, target_image)
            body_state = \
                body_recognizer.get_body_state(gray, depth_image, target_image)
            # print(pose_landmark_list)

            interval = time.time() - last_run
            estimated_fps = 1.0 / interval
            # print(estimated_fps)
            cv2.putText(target_image, f"FPS: {estimated_fps}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 2.0,
                        (255, 255, 255), 1, cv2.LINE_AA)

            if not demo_mode:
                target_image = np.hstack((target_image, depth_colored))

            cv2.imshow("Frame", target_image)
            last_run = time.time()

            # print(effective_gesture_texts)
            perc_state = ps.PerceptionState(
                people=[
                    ps.Person(
                        face=face_state,
                        hands=hand_states,
                        body=body_state,
                    )
                ]
            )

            # print(perc_state)
            data = ["PerceptionState".encode('utf-8'), perc_state.SerializeToString()]
            publisher.send_multipart(data)

            pressed_key = cv2.waitKey(3)
            if pressed_key == ord("a"):
                break

    finally:
        # ストリーミング停止
        pipeline.stop()


def main():
    cmd()


if __name__ == "__main__":
    main()
