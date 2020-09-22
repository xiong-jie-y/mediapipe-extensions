import ctypes
from multiprocessing import Process, Queue, Value
from pikapi.recognizers.base import PerformanceMeasurable
from pikapi.gui.visualize_gui import VisualizeGUI
from pikapi.logging import time_measure
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
import pikapi.logging
import pikapi.mediapipe_util as pmu

import pyrealsense2 as rs

from pikapi.utils.landmark import *
from pikapi.core.camera import IMUInfo

NOSE_INDEX = 4

# @jit(nopython=True)
def calculate_pose(denormalized_landmark: np.ndarray):
    """Calculate pose information in some forms."""
    # Calculate two of the head axis.
    center = np.mean(denormalized_landmark, axis=0)
    center_to_nose_direction = denormalized_landmark[NOSE_INDEX] - center
    up_direction = denormalized_landmark[9] - denormalized_landmark[164]

    def camera_coord_to_virtual_camera_coordinate(array):
        return np.array([-array[0], -array[1], -array[2]])

    # Transform the frame to the unity camera coordinate that has only translations of some unknown unit.
    # And create the frame so that it can be left-handed coordinate.
    up = camera_coord_to_virtual_camera_coordinate(up_direction / np.linalg.norm(up_direction))
    front = camera_coord_to_virtual_camera_coordinate(center_to_nose_direction / np.linalg.norm(center_to_nose_direction))
    right = np.cross(up, front)
    right = right / np.linalg.norm(right)

    rot = Rotation.align_vectors([right, up, front], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return rot, center_to_nose_direction, up_direction

class IntrinsicMatrix():
    def __init__(self, intrinsic_matrix):
        self.fx = intrinsic_matrix.fx
        self.fy = intrinsic_matrix.fy
        self.ppx = intrinsic_matrix.ppx
        self.ppy = intrinsic_matrix.ppy

class FaceGeometryRecognizer(PerformanceMeasurable):
    """Calculate geometrical properties of face.
    """

    def __init__(self, intrinsic_matrix):
        super().__init__()
        focal_length = intrinsic_matrix.fx
        self.runner = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/iris_tracking_gpu.pbtxt", [
                "face_landmarks_with_iris", "left_iris_depth_mm", "right_iris_depth_mm"],
            pmu.create_packet_map({"focal_length_pixel": focal_length})
        )
        self.intrinsic_matrix = intrinsic_matrix
        self.yes_or_no_estimator = YesOrNoEstimator()
        self.previous = None
        self.last_face_image = None

    # def _get_eye_direction_and_position(face_iris_landmark):
    #     import IPython; IPython.embed()
    #     left_center_iris = face_iris_landmark[468, :]
    #     right_center_iris = face_iris_landmark[468 + 5, :]

    def _adjust_by_imu(self, points: np.ndarray, imu_info: IMUInfo) -> np.ndarray:
        rot = get_shortest_rotvec_between_two_vector(
            np.array([0, -1.0, 0]), imu_info.acc)
        if rot is None:
            return points

        rotation_axis, theta = rot
        rot_sci = Rotation.from_rotvec(rotation_axis * theta)

        # import IPython; IPython.embed()
        return rot_sci.inv().apply(points)

    def nd_3d_to_nd_2d(self, nd_3d):
        return tuple([int(nd_3d[0]), int(nd_3d[1])])

    def _get_face_direction(self,
                            landmark_list, width, height, visualize_image, face_image
                            ):
        denormalized_landmark = get_denormalized_landmark_list(landmark_list, width, height)

        rot, center_to_nose_direction, up_direction = calculate_pose(denormalized_landmark)

        rot_vec = rot[0].as_rotvec()
        debug_txt = f"""Rotation
        {rot_vec}
        {np.linalg.norm(rot_vec)}
        {rot[1]}
        """

        cv2.line(visualize_image,
                 self.nd_3d_to_nd_2d(denormalized_landmark[NOSE_INDEX]),
                 self.nd_3d_to_nd_2d(
                     denormalized_landmark[NOSE_INDEX] + center_to_nose_direction * 4),
                 (255, 255, 255), 5)
        cv2.line(visualize_image,
                 self.nd_3d_to_nd_2d(denormalized_landmark[164]),
                 self.nd_3d_to_nd_2d(
                     denormalized_landmark[164] + up_direction * 4),
                 (255, 255, 255), 5)

        text_height = 1
        for line in debug_txt.split("\n"):
            cv2.putText(face_image, line, (0, text_height * 25), cv2.FONT_HERSHEY_PLAIN, 2.0,
                        (255, 255, 255), 1, cv2.LINE_AA)
            text_height += 1

        return rot_vec

    def _get_face_display_relation(self, center, imu_info):
        eye_camera_position = None
        camera_pose = None
        character_pose = None
        center_in_unity_pb = None

        # def array_to_vec(array, previous):
        if center is not None:
            SCREEN_TO_CHAR = 620

            # Calculate face center position in unity.
            face_center_in_unity = np.copy(center)
            face_center_in_unity = self._adjust_by_imu(
                np.array([face_center_in_unity]), imu_info)[0]

            # From camera coordinate to unity global coordinate.
            face_center_in_unity[0] = -face_center_in_unity[0]
            face_center_in_unity[1] = 848 - (face_center_in_unity[1] + 150)
            face_center_in_unity[2] += SCREEN_TO_CHAR
            face_center_in_unity[2] = -face_center_in_unity[2]
            center_in_unity_pb = ps.Vector(
                x=face_center_in_unity[0]/1000, y=face_center_in_unity[1]/1000, z=face_center_in_unity[2]/1000)
            # print("Center")
            # print(center_in_unity_pb)
            # print(face_center_in_unity)

            center = self._adjust_by_imu(np.array([center]), imu_info)[0]
            center[1] += 150
            # print(center)
            # print(center[2])
            screen_to_head = center[2]
            center[2] += SCREEN_TO_CHAR

            import math
            # cam_x = math.tan(math.atan2(center[0], center[2])) * center[0]
            cam_x = center[0] / center[2] * SCREEN_TO_CHAR
            cam_y = center[1] / center[2] * SCREEN_TO_CHAR
            camera_direction = np.array(
                [0, 848, -SCREEN_TO_CHAR]) - face_center_in_unity
            # camera_direction_in_unity = np.array([
            #     -camera_direction[0],
            #     -camera_direction[1],
            #     -camera_direction[2]
            #     ])
            # print(camera_direction)
            rot = get_shortest_rotvec_between_two_vector(
                np.array([0, 0, 1]), camera_direction)
            if rot is not None:
                # import IPython; IPython.embed()

                axis, theta = rot
                pose = Rotation.from_rotvec(axis * theta).as_quat()
                camera_pose = ps.Quaternion(
                    x=pose[0], y=pose[1], z=pose[2], w=pose[3])
                # char_pose = Rotation.from_rotvec(axis * theta).inv().as_quat()
                # print(center)
                xz_theta = math.atan2(-center[0], center[2])
                # print(xz_theta)
                char_pose = Rotation.from_rotvec(
                    np.array([0, 1, 0]) * -xz_theta).inv().as_quat()
                character_pose = ps.Quaternion(
                    x=char_pose[0], y=char_pose[1], z=char_pose[2], w=char_pose[3])
                # print(axis, theta)
                # print(character_pose)
                # import IPython; IPython.embed()

            eye_camera_position = ps.Vector(
                x=-cam_x/1000.0, y=-cam_y/1000.0, z=0.0)

            # print(eye_camera_position)
            # cam_y = math.tan(math.atan2(center[1], center[2])) * center[1]

        if center is None:
            vec = self.previous
        else:
            vec = ps.Vector(x=center[0], y=-center[1], z=center[2])
            self.previous = vec

        relation_to_monitor = ps.FaceMonitorRelation(
            eye_camera_position=eye_camera_position,
            eye_camera_pose=camera_pose, 
            character_pose=character_pose
        )
        return relation_to_monitor, center_in_unity_pb, vec

    def get_face_state(self,
                       rgb_image: np.ndarray, depth_image: np.ndarray,
                       visualize_image: np.ndarray, imu_info: IMUInfo) -> ps.Face:
        current_time = time.time()
        width = rgb_image.shape[1]
        height = rgb_image.shape[0]

        with self.time_measure("Run Face Graph"):
            self.runner.process_frame(rgb_image)

        with self.time_measure("Run Face Postprocess"):
            multi_face_landmarks = [self.runner.get_normalized_landmark_list(
                "face_landmarks_with_iris")]

            center = None
            center_to_nose_direction = None
            if len(multi_face_landmarks) != 0:
                face_landmark = np.array(multi_face_landmarks[0])
                from pikapi.logging import TimestampedData
                if len(face_landmark) != 0:
                    min_x = int(min(face_landmark[:, 0]) * width)
                    min_y = int(min(face_landmark[:, 1]) * height)
                    max_x = int(max(face_landmark[:, 0]) * width)
                    max_y = int(max(face_landmark[:, 1]) * height)
                    state = self.yes_or_no_estimator.get_state(
                        TimestampedData(current_time, multi_face_landmarks))
                    if state == 1:
                        state = "Yes"
                    elif state == -1:
                        state = "No"
                    else:
                        state = "No Gesture"

                    cv2.putText(visualize_image, state, (min_x, min_y), cv2.FONT_HERSHEY_PLAIN, 1.0,
                                (255, 255, 255), 1, cv2.LINE_AA)

                    # print(face_mesh_to_camera_coordinate(face_landmark, left_mm, right_mm))
                    mean_depth = depth_from_maybe_points_3d(
                        get_camera_coord_landmarks(face_landmark, width, height, depth_image, self.intrinsic_matrix))
                    # if mean_depth is None:

                    # import IPython; IPython.embed()
                    # center = get_face_center_3d(
                    #     face_landmark, left_mm, right_mm, width, height, self.intrinsic_matrix)
                    center = get_face_center_3d(
                        face_landmark, mean_depth, mean_depth, width, height, self.intrinsic_matrix)
                    # This center will have a y-axis down side.
                    # if center[2]:
                    # import IPython; IPython.embed()

                    x, y = project_point(
                        center, width, height, self.intrinsic_matrix)

                    # print(center)
                    # print(x,y)

                    # if mean_depth < 100:
                    #    import IPython; IPython.embed()

                    # mean depth
                    for i, point in enumerate(face_landmark):
                        cv2.circle(visualize_image, (int(point[0] * rgb_image.shape[1]), int(
                            point[1] * rgb_image.shape[0])), 3, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

                    rate = 8
                    face_width = max_x - min_x
                    face_height = max_y - min_y
                    face_image = np.zeros((face_height * rate, face_width * rate))
                    for i, point in enumerate(face_landmark):
                        draw_x = int((point[0] - min_x/width) * rate * width)
                        draw_y = int((point[1] - min_y/height) * rate * height)
                        cv2.circle(face_image, (draw_x, draw_y), 3,
                                (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                        cv2.putText(face_image, str(i), (draw_x, draw_y), cv2.FONT_HERSHEY_PLAIN, 1.0,
                                    (255, 255, 255), 1, cv2.LINE_AA)

                    with self.time_measure("Calc Face Direction"):
                        direction = self._get_face_direction(
                            face_landmark, width, height, visualize_image, face_image)
                    # direction = self._adjust_by_imu(
                    #     np.array([direction]), imu_info)[0]
                    center_to_nose_direction = ps.Vector(
                        x=direction[0], y=direction[1], z=direction[2])

                    self.last_face_image = face_image
                    # cv2.imshow("Face Image", face_image)

                    cv2.circle(visualize_image, (x, y), 3, (0, 0, 255),
                            thickness=-1, lineType=cv2.LINE_AA)
                    # print(np.mean([point[2] for point in face_landmark]))

            relation_to_monitor, center_in_unity, vec = self._get_face_display_relation(center, imu_info)

            # print("Face Direction")
            # print(center_to_nose_direction)
            return ps.Face(
                center=vec,
                center_in_unity=center_in_unity,
                pose=ps.FacePose(rotation_vector=center_to_nose_direction),
                relation_to_monitor=relation_to_monitor
            )


class FaceRecognizerProcess(Process):
    def __init__(self, rgb_image, depth_image, visualize_image, intrinsic_matrix):
        super(FaceRecognizerProcess, self).__init__()
        self.intrinsic_matrix = intrinsic_matrix
        self.queue = Queue()
        self.result_queue = Queue()
        self.latest_face_state = None
        self.finish_flag = Value(ctypes.c_bool, False)
        self.perf_queue = Queue()
        self.rgb_image = rgb_image
        self.depth_image=  depth_image
        self.visualize_image = visualize_image

    def run(self):
        # print("startfjkldjfdkjfdkfdjlkfdlsda")
        recognizer = FaceGeometryRecognizer(self.intrinsic_matrix)

        # pikapi.logging.manager_dict = md
        # pikapi.logging.manager = manager
        # print("startfjkldjfdkjfdkfdjlkfdlsda")

        latest_imu_info = None
        last_process = time.time()
        while not self.finish_flag.value:
            if (time.time() - last_process) < 0.040:
                continue

            if not self.queue.empty():
                task = self.queue.get()
                if task is None:
                    continue
                latest_imu_info = task

            if latest_imu_info is not None:
                last_process = time.time()
                face_state = recognizer.get_face_state(
                    np.frombuffer(memoryview(self.rgb_image), dtype=np.uint8).reshape((360, 640, 3)),
                    np.frombuffer(memoryview(self.depth_image), dtype=np.uint16).reshape((360, 640)),
                    np.frombuffer(memoryview(self.visualize_image), dtype=np.uint8).reshape((360, 640, 3)),
                    latest_imu_info
                    )
                self.result_queue.put(face_state.SerializeToString())
                

        self.perf_queue.put(dict(recognizer.time_measure_result))
        print("Finish")