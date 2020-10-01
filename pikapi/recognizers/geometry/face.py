import ctypes
from multiprocessing import Process, Queue, Value
from pikapi.recognizers.base import PerformanceMeasurable
from pikapi.gui.visualize_gui import VisualizeGUI
from pikapi.utils.logging import time_measure
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
import pikapi.utils.logging
import pikapi.mediapipe_util as pmu

import pyrealsense2 as rs

from pikapi.utils.landmark import *
from pikapi.core.camera import IMUInfo

NOSE_INDEX = 4
# NOSE_INDEX = 5
# NOSE_INDEX = 1

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


import openvino_open_model_zoo_toolkit.open_model_zoo_toolkit as omztk

def nd_3d_to_nd_2d(nd_3d):
    return tuple([int(nd_3d[0]), int(nd_3d[1])])


class FaceGeometryRecognizer(PerformanceMeasurable):
    """Calculate geometrical properties of face.
    """

    def __init__(self, intrinsic_matrix):
        super().__init__()
        focal_length = intrinsic_matrix.fx
        self.runner = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/iris_tracking_gpu.pbtxt", [
                "cloned_face_detections", "face_detections", "left_iris_depth_mm", "right_iris_depth_mm", "cloned_face_landmarks_with_iris",
                "cloned_left_eye_rect_from_landmarks", "cloned_right_eye_rect_from_landmarks"],
            pmu.create_packet_map({"focal_length_pixel": focal_length})
        )
        self.intrinsic_matrix = intrinsic_matrix
        self.yes_or_no_estimator = YesOrNoEstimator()
        self.previous = None
        self.last_face_image = None
        self.omz = omztk.openvino_omz()
        self.hp = self.omz.headPoseEstimator()
        self.emo = self.omz.emotionEstimator()
        self.gaze_estimator = self.omz.gazeEstimator()

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
                 nd_3d_to_nd_2d(denormalized_landmark[NOSE_INDEX]),
                 nd_3d_to_nd_2d(
                     denormalized_landmark[NOSE_INDEX] + center_to_nose_direction * 4),
                 (255, 255, 255), 5)
        cv2.line(visualize_image,
                 nd_3d_to_nd_2d(denormalized_landmark[164]),
                 nd_3d_to_nd_2d(
                     denormalized_landmark[164] + up_direction * 4),
                 (255, 255, 255), 5)

        text_height = 1
        for line in debug_txt.split("\n"):
            cv2.putText(face_image, line, (0, text_height * 25), cv2.FONT_HERSHEY_PLAIN, 2.0,
                        (255, 255, 255), 1, cv2.LINE_AA)
            text_height += 1

        return rot_vec, center_to_nose_direction, up_direction

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
            # print(face_center_in_unity)
            face_center_in_unity = self._adjust_by_imu(
                np.array([face_center_in_unity]), imu_info)[0]
            # print(face_center_in_unity)

            # From camera coordinate to unity global coordinate.
            face_center_in_unity[0] = -face_center_in_unity[0]
            face_center_in_unity[1] = 848 - (face_center_in_unity[1] - 300)
            face_center_in_unity[2] += SCREEN_TO_CHAR
            face_center_in_unity[2] = -face_center_in_unity[2]
            center_in_unity_pb = ps.Vector(
                x=face_center_in_unity[0]/1000, y=face_center_in_unity[1]/1000, z=face_center_in_unity[2]/1000)
            # print(center_in_unity_pb)
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

        # if center is None:
        #     vec = self.previous
        # else:
        #     vec = ps.Vector(x=center[0], y=-center[1], z=center[2])
        #     self.previous = vec

        relation_to_monitor = ps.FaceMonitorRelation(
            eye_camera_position=eye_camera_position,
            eye_camera_pose=camera_pose, 
            character_pose=character_pose
        )
        return relation_to_monitor, center_in_unity_pb

    def _get_face(self, face_landmark, width, height, rgb_image, depth_image, visualize_image, imu_info):
        center = None
        hand_center = None
        center_to_nose_direction = None
        front_vector = None
        up_vector = None
        if len(face_landmark) != 0:
            min_x = int(min(face_landmark[:, 0]) * width)
            min_y = int(min(face_landmark[:, 1]) * height)
            max_x = int(max(face_landmark[:, 0]) * width)
            max_y = int(max(face_landmark[:, 1]) * height)

            # print(face_mesh_to_camera_coordinate(face_landmark, left_mm, right_mm))
            # a = len(hand_landmark_points)
            # mean_depth = depth_from_maybe_points_3d(hand_landmark_points)
            # # if mean_depth is None:
            # assert a == len(hand_landmark_points)

            hand_center = pikapi.utils.landmark.get_3d_center(
                face_landmark, width, height, depth_image, self.intrinsic_matrix)
            if hand_center is None:
                return None
    
            mean_depth = hand_center[2]

            # print(hand_landmark_points)

            # print(depth_filter)
            # print(mean_depth)
            # import IPython; IPython.embed()
            # center = get_face_center_3d(
            #     face_landmark, left_mm, right_mm, width, height, self.intrinsic_matrix)
            if mean_depth is None:
                center = np.array([width, height, np.nan])
            else:
                center = get_face_center_3d(
                    face_landmark, mean_depth, mean_depth, width, height, self.intrinsic_matrix)
            # This center will have a y-axis down side.
            # if center[2]:
            # import IPython; IPython.embed()

                x, y = project_point(
                    center, width, height, self.intrinsic_matrix)
                cv2.circle(visualize_image, (x, y), 3, (0, 0, 255),
                        thickness=-1, lineType=cv2.LINE_AA)

            # print(center)
            # print(x,y)

            # print(np.mean(face_landmark, axis=0))

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
                direction, front_vector, up_vector = self._get_face_direction(
                    face_landmark, width, height, visualize_image, face_image)
            # direction = self._adjust_by_imu(
            #     np.array([direction]), imu_info)[0]
            center_to_nose_direction = ps.Vector(
                x=direction[0], y=direction[1], z=direction[2])

            self.last_face_image = face_image
            cv2.imshow("Face Image", face_image)
            cv2.waitKey(1) # For image to be shown.

            # print(np.mean([point[2] for point in face_landmark]))

        relation_to_monitor, center_in_unity = self._get_face_display_relation(hand_center, imu_info)

        if hand_center is None:
            vec = self.previous
        else:
            vec = ps.Vector(x=hand_center[0], y=hand_center[1], z=hand_center[2])
            self.previous = vec

        def numpy_to_vec(npa):
            if npa is None:
                return None
            return ps.Vector(x=npa[0], y=npa[1], z=npa[2])
        # print("Face Direction")
        # print(center_to_nose_direction)
        # assert (center_in_unity is None and vec is None) or (center_in_unity is not None or vec is not None)
        return ps.Face(
            center=vec,
            center_in_unity=center_in_unity,
            pose=ps.FacePose(
                rotation_vector=center_to_nose_direction, front_vector=numpy_to_vec(front_vector), up_vector=numpy_to_vec(up_vector)),
            relation_to_monitor=relation_to_monitor
        )

    def get_face_state(self,
                       rgb_image: np.ndarray, depth_image: np.ndarray,
                       visualize_image: np.ndarray, imu_info: IMUInfo) -> ps.Face:
        current_time = time.time()
        width = rgb_image.shape[1]
        height = rgb_image.shape[0]

        with self.time_measure("Run Face Graph"):
            self.runner.process_frame(rgb_image)

        with self.time_measure("Run Face Postprocess"):
            # print(multi_face_landmarks)
            # from pikapi.utils.logging import TimestampedData
            # state = self.yes_or_no_estimator.get_state(
            #     TimestampedData(current_time, multi_face_landmarks))
            # if state == 1:
            #     state = "Yes"
            # elif state == -1:
            #     state = "No"
            # else:
            #     state = "No Gesture"

            # cv2.putText(visualize_image, state, (min_x, min_y), cv2.FONT_HERSHEY_PLAIN, 1.0,
            #             (255, 255, 255), 1, cv2.LINE_AA)
            a = self.runner.get_normalized_landmark_list("cloned_face_landmarks_with_iris")
            face_detections = self.runner.get_proto_list("cloned_face_detections")
            left_eye_rect = self.runner.get_proto("cloned_left_eye_rect_from_landmarks")
            start = time.time()
            while face_detections is None or left_eye_rect is None:
                if (time.time() - start) > 0.020:
                    break
                self.runner.maybe_fetch()
                face_detections = self.runner.get_proto_list("cloned_face_detections")
                left_eye_rect = self.runner.get_proto("cloned_left_eye_rect_from_landmarks")
            # while a is None: #  or face_detections is None:
            #     self.runner.maybe_fetch()
            #     a = self.runner.get_normalized_landmark_list("face_landmarks_with_iris")
                # face_detections = self.runner.get_proto_list("face_detections")

            if face_detections is not None:
                from mediapipe.framework.formats.rect_pb2 import NormalizedRect
                rect = NormalizedRect()
                rect.ParseFromString(left_eye_rect)
                # print(rect)
                for detection in face_detections:
                    # detection = face_detections[0]
                    from mediapipe.framework.formats.detection_pb2 import Detection
                    n = Detection()
                    n.ParseFromString(detection)
                    box = n.location_data.relative_bounding_box
                    p1 = (int(box.xmin * width), int(box.ymin * height))
                    p2 = (int((box.xmin + box.width) * width), int((box.ymin + box.height) * height))

                    # print(p1)
                    # print(p2)
                    cv2.rectangle(visualize_image,p1, p2,(0,255,0),3)

                    import openvino_open_model_zoo_toolkit.open_model_zoo_toolkit as omztk
                    
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    face_img = omztk.ocv_crop(bgr_image, p1, p2, scale=1.8)
                    ypr = self.hp.run(face_img)
                    # print(yp)
                    print("YPR")
                    print(ypr)
                    rot_axises = np.array([[0,-1, 0], [1.0, 0, 0], [0, 0, -1]], dtype=np.float32)
                    rots = [Rotation.from_rotvec(ax * np.deg2rad(ad)) for ax, ad in zip(rot_axises, ypr)]
                    all_rot = rots[2] * rots[1] * rots[0]
                    # all_rot = Rotation.from_euler('xyz', ypr)
                    # print(all_rot.as_rotvec())
                    # all_rot = Rotation.from_rotvec([ypr[1], ypr[2], ypr[0]])
                    # all_rot = Rotation.from_euler()
                    # all_rot = Rotation.from_euler('YXZ', ypr)
                    face_rot_img = omztk.ocv_rotate(face_img, ypr[2])
                    emotion = self.emo.run(face_rot_img)
                    print(emotion)
                    base_frame = np.array([[0,-1, 0], [1.0, 0, 0], [0, 0, -1]], dtype=np.float32)
                    center_2d = np.array([0.0, 0, 10])
                    center_2dd  = tuple(project_point(center_2d, width, height, self.intrinsic_matrix))
                    print(center_2dd)
                    xyz = np.array([all_rot.apply(p * 2.0) + center_2d for p in base_frame])
    
                    print(self.gaze_estimator.infer_gaze_pose(rgb_image, , ypr))

                    points = [project_point(p, width, height, self.intrinsic_matrix) for p in xyz]
                    cv2.imshow("cropped", face_rot_img)
                    for p in points:
                        cv2.line(visualize_image, (p[0], p[1]), 
                            center_2dd,
                            (255, 255, 255), 5)
                    

            # print("dainyu")
            if a is not None:
                # multi_face_landmarks = [a]
                face_landmark = np.array(a)

                # while face_detections is None: #  or face_detections is None:
                #     self.runner.maybe_fetch()
                #     face_detections = self.runner.get_proto_list("face_rects_from_detections")

                # print("Face Det")
                # print(face_detections)
                # print(face_landmarks)
                return self._get_face(face_landmark, width, height, rgb_image, depth_image, visualize_image, imu_info)

class FaceRecognizerProcess(Process):
    def __init__(self, rgb_image, depth_image, visualize_image, intrinsic_matrix, new_image_ready_event):
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
        self.new_image_ready_event = new_image_ready_event

    def run(self):
        # print("startfjkldjfdkjfdkfdjlkfdlsda")
        recognizer = FaceGeometryRecognizer(self.intrinsic_matrix)

        # pikapi.utils.logging.manager_dict = md
        # pikapi.utils.logging.manager = manager
        # print("startfjkldjfdkjfdkfdjlkfdlsda")

        latest_imu_info = None
        last_process = time.time()
        while not self.finish_flag.value:
            # if (time.time() - last_process) < 0.040:
            #     continue

            if not self.queue.empty():
                task = self.queue.get()
                if task is None:
                    continue
                latest_imu_info = task

            if latest_imu_info is not None:
                last_process = time.time()
                self.new_image_ready_event.wait()
                face_state = recognizer.get_face_state(
                    np.frombuffer(memoryview(self.rgb_image), dtype=np.uint8).reshape((360, 640, 3)),
                    np.frombuffer(memoryview(self.depth_image), dtype=np.uint16).reshape((360, 640)),
                    np.frombuffer(memoryview(self.visualize_image), dtype=np.uint8).reshape((360, 640, 3)),
                    latest_imu_info
                    )
                self.new_image_ready_event.clear()
                if face_state is None:
                    self.result_queue.put(None)
                else:
                    self.result_queue.put(face_state.SerializeToString())
                

        self.perf_queue.put(dict(recognizer.time_measure_result))
        print("Finish")