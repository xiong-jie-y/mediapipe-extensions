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
import pikapi.mediapipe_util as pmu

import pyrealsense2 as rs


from pikapi.utils.landmark import *
from pikapi.core.camera import IMUInfo


class BodyGeometryRecognizer():
    """Calculates geometrical properties of body.
    """
    def __init__(self, intrinsic_matrix):
        self.pose_recognizer = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/upper_body_pose_tracking_gpu.pbtxt", [
                "pose_landmarks"], {})
        self.intrinsic_matrix = intrinsic_matrix

    def _proto_quaternion_from_rotation(self, rotation):
        quat = rotation.as_quat()
        return ps.Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

    def _get_bones(self, connections, landmark_list, width, height):
        denormalized_landmark_list = get_denormalized_landmark_list(landmark_list, width, height)

        # Not using z because z is unrealiable now.
        # TODO: Update it when model is accurate enough.
        denormalized_landmark_list[:, 2] = 0

        # rotations = get_relative_angles_from_finger_base(denormalized_landmark_list, connections)
        rotations = get_relative_angles_to_match_to_followee(denormalized_landmark_list, connections)

        # print(denormalized_landmark_list[15, :])

        bones = []
        names = ["shoulder", "arm"]
        for rotation, name in zip(rotations, names):
            quat = None
            if rotation is None:
                # quat = ps.Quaternion(x=0,y=0, z=0, w=1)
                quat = ps.Quaternion(x=0, y=0, z=0, w=0)
            else:
                axis, theta = rotation
                axis[0] = -axis[0]
                axis[1] = -axis[1]
                axis[2] = -axis[2]
                # quat = self._proto_quaternion_from_rotation(Rotation.from_rotvec(axis * -theta))
                r = axis * -theta
                quat = ps.Quaternion(x=r[0], y=r[1], z=r[2], w=0)
                # quat = self._proto_quaternion_from_rotation(Rotation.from_rotvec(axis * -theta))

            # print(name)
            # print(rotation)
            # print(axis)
            # pressed_key = cv2.waitKey(2)
            # if pressed_key == ord("b"):
            #     import IPython
            #     IPython.embed()
            bones.append(ps.Bone(pose=quat, name=name, z_angle=-theta))

        # print(bones)
        return bones

    def get_body_state(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                       visualize_image: np.ndarray):
        width = rgb_image.shape[1]
        height = rgb_image.shape[0]

        with time_measure("Run Body Graph"):
            self.pose_recognizer.process_frame(rgb_image)

        with time_measure("Run Body Postprocess"):
            pose_landmark_list = np.array(
                self.pose_recognizer.get_normalized_landmark_list("pose_landmarks"))

            bone_connections = {
                "right": [12, 11, 13, 15]
            }
            bones = []

            if len(pose_landmark_list) > 0:
                # if not is_too_far(pose_landmark_list, width, height, depth_image):
                mean_depth = depth_from_maybe_points_3d(
                    get_camera_coord_landmarks(pose_landmark_list, width, height, depth_image, self.intrinsic_matrix))
                if mean_depth > 1500:
                    for i, point in enumerate(pose_landmark_list):
                        cv2.circle(visualize_image, (int(point[0] * width), int(
                            point[1] * height)), 3, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                    return None

                for direction, connections in bone_connections.items():
                    new_bones = self._get_bones(connections, pose_landmark_list, width, height)
                    for new_bone in new_bones:
                        new_bone.name = f"{direction}_{new_bone.name}"
                    bones += new_bones
                for i, point in enumerate(pose_landmark_list):
                    cv2.circle(visualize_image, (int(point[0] * width), int(
                        point[1] * height)), 3, (255, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.putText(visualize_image, str(i), (int(point[0] * width), int(
                        point[1] * height)), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (255, 255, 255), 1, cv2.LINE_AA)
                # else:
                #     print(mean_depth)

            return ps.Body(
                bones=bones
            )

    def get_state(self, *args, result={}):
        result['body_state'] = self.get_body_state(*args)