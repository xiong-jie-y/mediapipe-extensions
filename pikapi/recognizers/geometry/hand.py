"""This is the perception server that accept RGBD camera input 
and output data through ipc (currently zeromq) in the proto message format..
"""
import contextlib
import ctypes
from multiprocessing import Process, Queue, Value
from pikapi.recognizers.base import PerformanceMeasurable
# from pikapi.logging import time_measure
from numpy.core.fromnumeric import mean
from pikapi.utils.unity import realsense_vec_to_unity_char_vec
import time

from scipy.spatial.transform.rotation import Rotation
import pikapi.protos.perception_state_pb2 as ps
from pikapi.head_gestures import YesOrNoEstimator
import logging
from typing import ContextManager, Dict, List
import IPython
import cv2
import numpy as np

from pikapi import graph_runner

import click

import pikapi
import pikapi.utils.logging
import pikapi.mediapipe_util as pmu

import pyrealsense2 as rs


from pikapi.utils.landmark import *
from pikapi.core.camera import IMUInfo

time_measure = None

class HandGestureRecognizer(PerformanceMeasurable):
    """Calculate geometrical properties of hand.
    """
    def __init__(self, intrinsic_matrix):
        super().__init__()
        self.hand_recognizer = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/multi_hand_tracking_gpu.pbtxt", [
                "multi_hand_landmarks", "multi_handedness"], {})
        self.logger = logging.getLogger(__class__.__name__)

        # TODO: Change to camera object
        self.intrinsic_matrix = intrinsic_matrix

        self.record_trajectory = False
        self.trajectory = []
        

    def _get_achimuitehoi_gesture(self, landmark_list: np.ndarray,
                                  width: int, height: int, visualize_image: np.ndarray, min_x: int, min_y: int) -> str:
        landmark_list = get_denormalized_landmark_list(
            landmark_list, width, height)

        max_angles = {}
        for finger_name in FINGER_IDS.keys():
            rotations = get_relative_angles_from_finger_base(
                landmark_list, FINGER_IDS[finger_name])
            max_angles[finger_name] = np.max([rot[1] for rot in rotations])

        closes = ['middle_finger', 'ring_finger', 'pinky_finger']
        opens = ['index_finger']

        # OPEN_THRESH = 4/np.pi
        OPEN_THRESH = np.pi / 4
        pointing_something = True

        for finger_name in opens:
            if max_angles[finger_name] > OPEN_THRESH:
                pointing_something = False

        for finger_name in closes:
            if max_angles[finger_name] < OPEN_THRESH:
                pointing_something = False

        index_fingers = landmark_list[FINGER_IDS['index_finger']]
        pointing_dir = index_fingers[-1] - index_fingers[0]
        pointing_dir = pointing_dir / np.linalg.norm(pointing_dir)

        direction = None
        if abs(pointing_dir[2]) > abs(pointing_dir[0]) and \
                abs(pointing_dir[2]) > abs(pointing_dir[1]):
            direction = "camera"
        else:
            # TODO: Need to fix
            if (abs(pointing_dir[0]) < 0.07 and abs(pointing_dir[1]) < 0.07):
                direction = "camera"
            else:
                if abs(pointing_dir[1]) > abs(pointing_dir[0]):
                    # left from the camera
                    if pointing_dir[1] < -0.7:
                        direction = "up"
                    elif pointing_dir[1] > 0.7:
                        direction = "down"
                    else:
                        direction = "ambigous"
                else:
                    if pointing_dir[0] < -0.7:
                        direction = "left"
                    elif pointing_dir[0] > 0.7:
                        direction = "right"
                    else:
                        direction = "ambiguous"

        assert direction is not None

        state_tag = f"pointing_to_{direction}" if pointing_something else "not_pointing"
        cv2.putText(visualize_image,
                    state_tag,
                    (min_x, min_y), cv2.FONT_HERSHEY_PLAIN, 2.0,
                    (0, 0, 255), 3, cv2.LINE_AA)

        return state_tag

    def _get_finger_rotations(self, landmark_list, width, height) -> Dict[str, Rotation]:
        landmark_list = get_denormalized_landmark_list(
            landmark_list, width, height)

        finger_name_rotations_map = {}
        for finger_name in FINGER_IDS.keys():
            rotations = get_relative_angles_from_xy_plain(
                landmark_list[FINGER_IDS[finger_name]])
            finger_name_rotations_map[finger_name] = [
                Rotation.from_rotvec(axis * theta) for axis, theta in rotations]

        return finger_name_rotations_map

    def _proto_quaternion_from_rotation(self, rotation):
        quat = rotation.as_quat()
        return ps.Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

    def _get_fingers(self, denormalized_landmark_list, palm_rotation) -> List[ps.Finger]:        
        # Python version.
        # palm_rotation = get_palm_angle(denormalized_landmark_list)

        direction_normalized_landmark_list = None
        # It is already normalized, when None.
        if palm_rotation is None:
            direction_normalized_landmark_list = denormalized_landmark_list
        else:
            direction_normalized_landmark_list = palm_rotation.inv().apply(denormalized_landmark_list)

        # assert(direction_normalized_landmark_list is not None)

        finger_states = []
        for finger_name in FINGER_IDS.keys():
            rotations = get_relative_angles_from_xy_plain(direction_normalized_landmark_list[FINGER_IDS[finger_name]])
            finger_states.append(
                ps.Finger(finger_name=finger_name, 
                # rotations=[
                #     self._proto_quaternion_from_rotation(
                #         Rotation.from_rotvec(realsense_vec_to_unity_char_vec(axis) * theta)) for axis, theta in rotations
                # ], 
                rotation_angles=[
                    theta for axis, theta in rotations
            ]
                )
            )

        # C++ version.
        # finger_name_to_rotations = pikapi.landmark_utils.get_fingers(denormalized_landmark_list, FINGER_IDS)

        # finger_states = []
        # for finger_name, rotations in finger_name_to_rotations.items():
        #     finger_states.append(
        #         ps.Finger(finger_name=finger_name, 
        #         # rotations=[
        #         #     self._proto_quaternion_from_rotation(
        #         #         Rotation.from_rotvec(realsense_vec_to_unity_char_vec(axis) * theta)) for axis, theta in rotations
        #         # ], 
        #         rotation_angles=[
        #             theta for axis, theta in rotations
        #         ]
        #         )
        #     )

        return finger_states

    def _accumulate_trajectory(self, landmark_list: np.ndarray,
                               width: int, height: int, visualize_image: np.ndarray):
        # pressed_key = cv2.waitKey(2)
        # if pressed_key == ord("t"):
        #     self.logger.info("Start Logging")
        #     self.trajectory = []
        # self.record_trajectory = True

        if self.record_trajectory:
            self.trajectory.append(
                landmark_list[FINGER_IDS['index_finger']][-1])

            for first, second in zip(self.trajectory[:-1], self.trajectory[1:]):
                cv2.line(visualize_image, (int(first[0] * width), int(
                    first[1] * height)), (int(second[0] * width), int(
                        second[1] * height)), (255, 255, 255), 5)

    def get_hand_states(self,
                        rgb_image: np.ndarray, depth_image: np.ndarray,
                        visualize_image: np.ndarray) -> List[str]:
        width = rgb_image.shape[1]
        height = rgb_image.shape[0]

        with self.time_measure("Run Hand Graph"):
        # Estimate hand landmarks and gesture text
            self.hand_recognizer.process_frame(rgb_image)
        hand_landmarks = None
        multi_handedness = None
        with self.time_measure("Get Hand Landmark Result"):
            hand_landmarks = \
                self.hand_recognizer.get_normalized_landmark_lists("multi_hand_landmarks")
            multi_handedness = self.hand_recognizer.get_proto_list("multi_handedness")
            # gesture_texts = self.hand_recognizer.get_string_array("gesture_texts")
            # if hand_landmarks is None:
            #     hand_landmarks = np.array([])
            # else:
            #     hand_landmarks = np.array(hand_landmarks)
            # if multi_handedness is None:
            #     multi_handedness = []

            # print(hand_landmarks)
            while hand_landmarks is None or multi_handedness is None:
                self.hand_recognizer.maybe_fetch()
                hand_landmarks = self.hand_recognizer.get_normalized_landmark_lists("multi_hand_landmarks")
                multi_handedness = self.hand_recognizer.get_proto_list("multi_handedness")

            hand_landmarks=  np.array(hand_landmarks)
            # print(hand_landmarks)
            # print(self.hand_recognizer.get_normalized_landmark_lists("multi_hand_landmarks"))
            # print(multi_handedness)
            from mediapipe.framework.formats.classification_pb2 import ClassificationList
            multi_handedness_parsed = []
            for handedness in multi_handedness:
                a = ClassificationList()
                a.ParseFromString(handedness)
                # Because it's reverse.
                if a.classification[0].label == "Right":
                    a.classification[0].label = "Left"
                else:
                    a.classification[0].label = "Right"

                multi_handedness_parsed.append(a)
                # import IPython; IPython.embed()

        with self.time_measure("Run Hand Postprocess"):
            hand_states = []
            # assert len(hand_landmarks) == len(gesture_texts)
            acchimuitehoi_gesture_name = None
            for hand_landmark_list, handedness in zip(hand_landmarks, multi_handedness_parsed):
                effective_gesture_texts = []
                hand_landmark_points = get_camera_coord_landmarks(
                    hand_landmark_list, width, height, depth_image, self.intrinsic_matrix)
                # Filter conversing person's hand.
                zs = hand_landmark_points[:, 2]

                nan_filter = ~np.isnan(zs)
                zs = zs[nan_filter]
                if len(zs) == 0:
                    for i, point in enumerate(hand_landmark_list):
                        cv2.circle(visualize_image, (int(point[0] * width), int(
                            point[1] * height)), 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
                    continue
                # inlier_filter = (zs < np.percentile(zs, 70)) & (zs > np.percentile(zs, 30))
                inlier_filter = (zs < np.percentile(zs, 10))
                zs = zs[inlier_filter]

                if len(zs) == 0:
                    for i, point in enumerate(hand_landmark_list):
                        cv2.circle(visualize_image, (int(point[0] * width), int(
                            point[1] * height)), 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
                    continue
                mean_depth = np.mean(zs)
                # hand_center_xy = np.mean(hand_landmark_points[nan_filter][inlier_filter][:, [0, 1]], axis=0)
                # hand_center = np.concatenate((hand_center_xy, [mean_depth]))
                
                # Use wrist for more stability.
                hand_center = np.concatenate((hand_landmark_points[0], [mean_depth]))

                min_x = int(min(hand_landmark_list[:, 0]) * width)
                min_y = int(min(hand_landmark_list[:, 1]) * height)
                max_x = int(max(hand_landmark_list[:, 0]) * width)
                max_y = int(max(hand_landmark_list[:, 1]) * height)

                if mean_depth > 1500:
                    for i, point in enumerate(hand_landmark_list):
                        cv2.circle(visualize_image, (int(point[0] * width), int(
                            point[1] * height)), 3, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.putText(visualize_image, str(mean_depth), (min_x, min_y + 50), cv2.FONT_HERSHEY_PLAIN, 1.0,
                            (0, 0, 0), 2, cv2.LINE_AA)
                    continue

                # if mean_depth > 1000:
                #     print(hand_landmark_points[:, 2])

                # print(hand_landmark_points)

                # print(np.mean(hand_landmark_list, axis=0))

                # print(zs)
                print(f"hand", mean_depth)

                # print(zs)
                # print(mean_depth)
                # print(hand_center)
                # self.logger.info(f"In a range: {mean_depth}")

                import pikapi.utils.opencv
                pikapi.utils.opencv.overlay_rect_with_opacity(
                    visualize_image, (min_x, min_y, (max_x - min_x), (max_y - min_y)))
                # effective_gesture_texts.append(gesture_text)
                visualize_landmark_list(
                    hand_landmark_list, width, height, visualize_image)

                # cv2.putText(visualize_image, gesture_text, (min_x, min_y + 25), cv2.FONT_HERSHEY_PLAIN, 1.0,
                #             (0, 0, 0), 2, cv2.LINE_AA)
                # print(handedness.classification[0].label)
                cv2.putText(visualize_image, handedness.classification[0].label, (min_x, min_y + 100), cv2.FONT_HERSHEY_PLAIN, 1.0,
                            (0, 0, 0), 2, cv2.LINE_AA)

                with self.time_measure("Recognize Acchimuitehoi Gesture"):
                    effective_gesture_texts.append(self._get_achimuitehoi_gesture(
                        hand_landmark_list, width, height, visualize_image, min_x, min_y + 50))
                with self.time_measure("Calc finger rotations"):
                    denormalized_landmark_list = get_denormalized_landmark_list(
                        hand_landmark_list, width, height)
                    paxis, pangle = pikapi.landmark_utils.estimate_palm_rotation(denormalized_landmark_list, handedness.classification[0].label)
                    palm_rotation = Rotation.from_rotvec(paxis * pangle)
                    fingers = self._get_fingers(
                            denormalized_landmark_list, palm_rotation)
                    palm_rotvec = paxis * pangle
                    # print(paxis, pangle)
                self._accumulate_trajectory(
                    hand_landmark_list, width, height, visualize_image)

                # print(hand_center)
                hand_states.append(ps.Hand(
                    gesture_names=effective_gesture_texts,
                    center=ps.Vector(x=hand_center[0], y=hand_center[1], z=hand_center[2]),
                    palm_rotation=ps.Vector(x=palm_rotvec[0], y=palm_rotvec[1],z= palm_rotvec[2]),
                    hand_exist_side=handedness.classification[0].label,
                    fingers=fingers))

        # print(hand_states)
        return hand_states
    def get_state(self, *args, result={}):
        result['hand_states'] = self.get_hand_states(*args)

class HandRecognizerProcess(Process):
    def __init__(self, rgb_image, depth_image, visualize_image, intrinsic_matrix, new_image_ready_event):
        super(HandRecognizerProcess, self).__init__()
        self.intrinsic_matrix = intrinsic_matrix
        self.queue = Queue()
        self.result_queue = Queue()
        self.finish_flag = Value(ctypes.c_bool, False)
        self.really_finish_flag = Value(ctypes.c_bool, False)
        self.perf_queue = Queue()
        self.latest_face_state = None
        self.rgb_image = rgb_image
        self.depth_image=  depth_image
        self.visualize_image = visualize_image
        self.new_image_ready_event = new_image_ready_event
        # self.md = md
        # self.manager = manager
        # self.dict = {}
    
    def run(self):
        hand_gesture_recognizer = HandGestureRecognizer(self.intrinsic_matrix)
        # pikapi.logging.manager_dict = self.md
        # pikapi.logging.manager = self.manager

        imu_info = None
        last_processed = time.time()
        while not self.finish_flag.value:
            # if (time.time() - last_processed) < 0.020:
            #     continue
    
            # print("ahoooo")
            if not self.queue.empty():
                task = self.queue.get()
                if task is None:
                    continue

                # print(time.time() - last_processed)
                # if (time.time() - last_processed) < 0.0013:
                #     continue
                imu_info = task

            if imu_info is not None:
                last_processed = time.time()
                self.new_image_ready_event.wait()
                ss = hand_gesture_recognizer.get_hand_states(
                    np.frombuffer(memoryview(self.rgb_image), dtype=np.uint8).reshape((360, 640, 3)),
                    np.frombuffer(memoryview(self.depth_image), dtype=np.uint16).reshape((360, 640)),
                    np.frombuffer(memoryview(self.visualize_image), dtype=np.uint8).reshape((360, 640, 3)),
                    )
                self.new_image_ready_event.clear()
                # self.result_queue.put(100)
                self.result_queue.put([s.SerializeToString() for s in ss])
                # self.perf_queue.put(1000)

        print("Finished")
        self.perf_queue.put(dict(hand_gesture_recognizer.time_measure_result))
        # while not self.really_finish_flag:
        #     pass