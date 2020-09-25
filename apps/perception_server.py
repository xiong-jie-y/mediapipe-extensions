"""This is the perception server that accept RGBD camera input 
and output data through ipc (currently zeromq) in the proto message format..
"""
import ctypes
import multiprocessing
from multiprocessing import Queue
import os
from pikapi.gui.visualize_gui import VisualizeGUI, create_visualize_gui_manager
from pikapi.logging import create_logger_manager, create_mp_logger, time_measure
from threading import Thread
import threading
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
from pikapi.recognizers.geometry.face import FaceGeometryRecognizer, FaceRecognizerProcess, IntrinsicMatrix
from pikapi.recognizers.geometry.hand import HandGestureRecognizer, HandRecognizerProcess
from pikapi.recognizers.geometry.body import BodyGeometryRecognizer


@click.command()
@click.option('--camera-id', '-c', default=0, type=int)
@click.option('--run-name', default="general")
@click.option('--relative-depth', is_flag=True)
@click.option('--use-realsense', is_flag=True)
@click.option('--demo-mode', is_flag=True)
@click.option('--gui-single-process', is_flag=True)
@click.option('--perception-single-process', is_flag=True)
def cmd(
        camera_id, run_name, relative_depth, use_realsense,
        demo_mode, gui_single_process, perception_single_process):
    intrinsic_matrix = get_intrinsic_matrix(camera_id)
    # Reason is unknown, but waiting at some point is necessary.
    # less than 0.1 was not enough from experiment.
    time.sleep(0.7)

    multiprocessing.set_start_method('spawn')

    # body_recognizer = BodyGeometryRecognizer(intrinsic_matrix)
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

    WIDTH = 640
    HEIGHT = 360

    # Should be created with manager when parallel.
    manager = None
    if gui_single_process:
        visualizer = VisualizeGUI(width=WIDTH, height=HEIGHT,
                                  run_multiprocess=not gui_single_process, demo_mode=demo_mode)
    else:
        shared_gray = multiprocessing.sharedctypes.RawArray('B', 360 * 640*3)
        # manager = create_visualize_gui_manager()
        # visualizer = manager.VisualizeGUI(width=WIDTH, height=HEIGHT, demo_mode=demo_mode)
        visualizer = VisualizeGUI(width=WIDTH, height=HEIGHT, demo_mode=demo_mode)
        # vis_server = manager.VisServer(640, 360)

        target_image_buf = visualizer.get_image_buf_internal()
        depth_image_shared = visualizer.get_depth_image_buf_internal()

    # queue = Queue()
    # result_queue = Queue()

    # cap = cv2.VideoCapture(camera_id)
    # while True:
    #     ret, frame = cap.read()
    #     if frame is None:
    #         break
        
    #     cv2.imshow("Test", frame)
    #     cv2.waitKey(3)
    # last_shot = time.time()
    # while True:
    #     frames = pipeline.wait_for_frames()
    #     aligned_frames = align.process(frames)
    #     color_frame = aligned_frames.get_color_frame()
    #     depth_frame = aligned_frames.get_depth_frame()
    #     frame = np.asanyarray(color_frame.get_data())
    #     cv2.imshow("Test", frame)
    #     if time.time() - last_shot > 2:
    #         cv2.imwrite('lena_opencv_red.jpg', frame)
    #     cv2.waitKey(3)

    # manager, manager_dict = create_mp_logger()
    # logger_manager = create_logger_manager()
    # perf_logger = logger_manager.PerfLogger()
    # face_processor = multiprocessing.Process(target=face_recognize,
    #                                          args=(
    #                                              queue, result_queue, shared_gray, depth_image_shared, 
    #                                              target_image_buf, IntrinsicMatrix(intrinsic_matrix)
    #                                              ))
    face_processor = FaceRecognizerProcess(shared_gray, depth_image_shared,
                                           target_image_buf, IntrinsicMatrix(intrinsic_matrix),
                                           visualizer.new_image_ready_event
                                           )
    face_processor.start()

    hand_processor = HandRecognizerProcess(shared_gray, depth_image_shared,
                                           target_image_buf, IntrinsicMatrix(intrinsic_matrix),
                                           visualizer.new_image_ready_event
                                           )
    hand_processor.start()

    import pikapi.logging
    last_run = time.time()
    from collections import deque
    acc_vectors = deque([])

    latest_hand_states = None
    latest_face_state = None

    frame_times = []
    try:
        while(True):
            # print( (time.time() - last_run))
            if (time.time() - last_run) < 0.013:
                continue

            with time_measure("Frame Fetch"):
                last_run = time.time()
                current_time = time.time()
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame or not color_frame:
                    continue

                # import IPython; IPython.embed()

            with time_measure("IMU Calculation"):
                # Camera pose estimation.
                acc = frames[2].as_motion_frame().get_motion_data()
                gyro = frames[3].as_motion_frame().get_motion_data()
                timestamp = frames[3].as_motion_frame().get_timestamp()
                # rotation_estimator.process_gyro(
                #     np.array([gyro.x, gyro.y, gyro.z]), timestamp)
                # rotation_estimator.process_accel(np.array([acc.x, acc.y, acc.z]))
                # theta = rotation_estimator.get_theta()
                acc_vectors.append(np.array([acc.x, acc.y, acc.z]))
                if len(acc_vectors) > 200:
                    acc_vectors.popleft()
                acc = np.mean(acc_vectors, axis=0)
                imu_info = IMUInfo(acc)

                # print(acc)
                # print(np.array([acc.x, acc.y, acc.z]))

                # print(theta)

            with time_measure("Frame Preparation"):
                # frame = np.asanyarray(color_frame.get_data())
                # depth_image = np.asanyarray(depth_frame.get_data())

                frame = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape(360, 640, 3)
                depth_image = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(360, 640)

                # import IPython; IPython.embed()
                

                # depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(
                #     depth_image, alpha=0.08), cv2.COLORMAP_JET)

                # cv2.imshow("orig", frame)
                # cv2.waitKey(2)

                gray = np.frombuffer(memoryview(shared_gray), dtype=np.uint8).reshape((360, 640, 3))
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, gray)
                # memoryview(shared_gray).cast('B')[:] = memoryview(gray).cast('B')[:]
                # gray_shared = np.frombuffer(memoryview(shared_gray), dtype=np.uint8).reshape((360, 640, 3))

            with time_measure("Pass Image"):
                # import IPython; IPython.embed()
                visualizer.pass_image(frame, depth_image)

                # ret, frame = cap.read()
                # if frame is None:
                #     break

                # import IPython; IPython.embed()
                # frame[depth_image > 2500] = 0
                # frame[depth_image == 0] = 0
                # depth_image[depth_image > 2500] = 0
                # depth_image[depth_image == 0] = 0

            # with time_measure("Pass Images"):
            #     memoryview(vis_server.get_shared_memory()).cast('B')[:] = memoryview(frame).cast('B')[:]
                # vis_server.pass_image(frame, depth_image)
                # depth_colored = cv2.convertScaleAbs(depth_image, alpha=0.08)

            if demo_mode:
                target_image = depth_colored
            else:
                target_image = visualizer.get_image_buf()

            with time_measure("Perception"):
                if perception_single_process:
                    run_on_threads = False
                    if run_on_threads:
                        result = {}
                        face_th = threading.Thread(target=face_recognizer.get_state,
                                                   args=(gray, depth_image, target_image, imu_info), kwargs={"result": result})
                        face_th.start()
                        hand_th = threading.Thread(target=hand_gesture_recognizer.get_state,
                                                   args=(gray, depth_image, target_image), kwargs={"result": result})
                        hand_th.start()
                        body_th = threading.Thread(target=body_recognizer.get_state,
                                                   args=(gray, depth_image, target_image), kwargs={"result": result})
                        body_th.start()

                        face_th.join()
                        hand_th.join()
                        body_th.join()

                        face_state = result['face_state']
                        hand_states = result['hand_states']
                        body_state = result['body_state']
                    else:
                        face_state = face_recognizer.get_face_state(
                            gray, depth_image, target_image, imu_info)

                        hand_states = \
                            hand_gesture_recognizer.get_hand_states(
                                gray, depth_image, target_image)
                        body_state = \
                            body_recognizer.get_body_state(gray, depth_image, target_image)
                        # print(pose_landmark_list)
                else:
                    result = {}
                    # face_processor.queue.put((shared_gray, depth_image_shared, target_image_buf, imu_info))
                    face_processor.queue.put(imu_info)
                    hand_processor.queue.put(imu_info)

                    # if result_queue.get():
                    #     pass

                    # if hand_processor.result_queue.get():
                    #     pass

                    # import IPython; IPython.embed()

                    # face_th = multiprocessing.Process(target=face_recognizer.get_state, \
                    #         args=(shared_gray, depth_image_shared, target_image_buf, imu_info), kwargs={"result": result})
                    # face_th.start()
                    # hand_th = multiprocessing.Process(target=hand_gesgture_recognizer.get_state, \
                    #         args=(gray_shared, depth_image_shared, taret_image), kwargs={"result": result})
                    # hand_th.start()
                    # body_th = multiprocessing.Process(target=body_recognizer.get_state, \
                    #         args=(gray_shared, depth_image_shared, target_image), kwargs={"result": result})
                    # body_th.start()

                    # face_th.join()
                    # hand_th.join()
                    # body_th.join()

                    # face_state = result['face_state']
                    # hand_states = result['hand_states']
                    # body_state = result['body_state']

            interval = time.time() - last_run
            estimated_fps = 1.0 / interval
            # print(estimated_fps)
            frame_times.append(interval * 1000)
            cv2.putText(target_image, f"Frame Time{interval * 1000}[ms]", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2.0,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(target_image, f"FPS: {estimated_fps}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 2.0,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # print("CopyImage", np.mean(pikapi.logging.time_measure_result["CopyImage"]))
            # print("CreateStack", np.mean(pikapi.logging.time_measure_result["Create Stack"]))

            TIMEOUT_S = 0.001
            # got_new_result = False

            # Wait both result. Just want to run paralle.
            # if not face_processor.result_queue.empty():
            # time.sleep(0.001)
            # result = None
            # while not face_processor.result_queue.empty():
            #     # result = face_processor.result_queue.get(timeout=0.001)
            #     result = face_processor.result_queue.get(False)
            # if result is not None:
            #     latest_face_state = ps.Face()
            #     latest_face_state.ParseFromString(result)
            # got_new_result = True
            result = face_processor.result_queue.get()
            latest_face_state = ps.Face()
            latest_face_state.ParseFromString(result)

            # if not hand_processor.result_queue.empty():
            # result = None
            # while not hand_processor.result_queue.empty():
            #     # result = hand_processor.result_queue.get(timeout=0.001)
            #     result = hand_processor.result_queue.get(False)
            # if result is not None:
            #     latest_hand_states = []
            #     for r in result:
            #         hand = ps.Hand()
            #         hand.ParseFromString(r)
            #         latest_hand_states.append(hand)
            # got_new_result = True
            result = hand_processor.result_queue.get()
            latest_hand_states = []
            for r in result:
                hand = ps.Hand()
                hand.ParseFromString(r)
                latest_hand_states.append(hand)

            # if got_new_result:
            visualizer.draw()

            # if face_recognizer.last_face_image is not None:
            #     cv2.imshow("Face Image", face_recognizer.last_face_image)
            last_run = time.time()

            # import IPython; IPython.embed()

            # print(effective_gesture_texts)
            perc_state = ps.PerceptionState(
                people=[
                    ps.Person(
                        face=latest_face_state,
                        hands=latest_hand_states,
                        # body=None,
                    )
                ]
            )

            # print(perc_state)
            data = ["PerceptionState".encode('utf-8'), perc_state.SerializeToString()]
            publisher.send_multipart(data)

            if visualizer.end_issued():
                import json
                # import IPython
                # IPython.embed()

                perf_dict = dict(pikapi.logging.time_measure_result)
                perf_dict['frame_ms'] = frame_times

                face_processor.finish_flag.value = True
                hand_processor.finish_flag.value = True

                print("finish requested")
                hand_perf = hand_processor.perf_queue.get()
                if hand_perf:
                    for key, val in hand_perf.items():
                        perf_dict[key] = val
                face_perf = face_processor.perf_queue.get()
                if face_perf:
                    for key, val in face_perf.items():
                        perf_dict[key] = val
                # hand_processor.really_finish_flag.value = True

                json.dump(perf_dict, open(
                    f"{run_name}_performance.json", "w"))

                hand_processor.kill()
                face_processor.kill()
                    
                break
    finally:
        print("Stopping Realsense.")
        # ストリーミング停止
        pipeline.stop()


def main():
    cmd()


if __name__ == "__main__":
    main()
