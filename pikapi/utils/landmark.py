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

# def face_mesh_to_camera_coordinate(face_iris_landmark, left_mm, right_mm):
#     import IPython; IPython.embed()
#     left_center_iris = face_iris_landmark[468, :]
#     right_center_iris = face_iris_landmark[468 + 5, :]

#     left_relatives = np.copy(face_iris_landmark)
#     left_relatives[:, 2] =  left_relatives[:, 2] - left_center_iris[2] + left_mm
#     left_relatives[:, 2] =  left_relatives[:, 2] - left_center_iris[2] + left_mm


# def get_3d_landmark_list_from_depth(landmark_list, width, height, depth_image, intrinsic_matrix):
#     landmark_list_copy = np.copy(landmark_list)
#     landmark_list_copy[:, 0] *= width
#     landmark_list_copy[:, 1] *= height
from numba import jit

@pikapi.logging.save_argument
def get_3d_center(face_landmark, width, height, depth_image, intrinsic_matrix):
    hand_landmark_points = get_camera_coord_landmarks(face_landmark, width, height, depth_image, intrinsic_matrix)
    zs = hand_landmark_points[:, 2]
    nan_filter = ~np.isnan(zs)
    zs = zs[nan_filter]
    if len(zs) == 0:
        return None
    inlier_filter = (zs < np.percentile(zs, 90)) & (zs > np.percentile(zs, 10))
    zs = zs[inlier_filter]
    if len(zs) == 0:
        return None

    mean_depth = np.mean(zs)
    hand_center_xy = np.mean(hand_landmark_points[nan_filter][inlier_filter][:, [0, 1]], axis=0)
    hand_center = np.concatenate((hand_center_xy, [mean_depth]))

    return hand_center

# @jit(nopython=True)
# @pikapi.logging.save_argument
def get_camera_coord_landmarks_numba(
        normalized_landmark_list: np.ndarray, width: int, height: int,
        depth_image: np.ndarray, ppx, ppy, fx, fy):
    points_c = np.zeros_like(normalized_landmark_list)
    for i in range(0, len(normalized_landmark_list)):
        point = normalized_landmark_list[i]
        # These points are outside of the depth images.
        if point[0] < 0.0 or point[0] > 1.0 or point[1] < 0.0 or point[1] > 1.0:
            # points_c.append([point[0], point[1], None])
            points_c[i][0] = point[0]
            points_c[i][1] = point[1]
            points_c[i][2] = np.nan
            continue

        x_pix = int(point[0] * width)
        y_pix = int(point[1] * height)
        depth_value = depth_image[y_pix, x_pix]

        # This depth value is invalid.
        if depth_value == 0:
            points_c[i][0] = point[0]
            points_c[i][1] = point[1]
            points_c[i][2] = np.nan
            continue

        # print(x_pix, ppx)
        # print(y_pix, ppy)
        # Get 3d coordinate from depth.
        x = depth_value * (x_pix - ppx) / fx
        y = depth_value * (y_pix - ppy) / fy
        # points_c.append([x, y, depth_value])
        points_c[i][0] = x
        points_c[i][1] = y
        points_c[i][2] = depth_value

    # assert len(points_c) == len(normalized_landmark_list)

    # return np.array(points_c, dtype=np.float)
    return points_c


def get_camera_coord_landmarks(
        normalized_landmark_list: np.ndarray, width: int, height: int,
        depth_image: np.ndarray, intrinsic_matrix: np.ndarray):
    """Get landmark list in camera coordinate.

    Returns:
        The 3d points in camera coord in [x,y,z].
    """
    with time_measure("GetCameraCoordinate"):
        return get_camera_coord_landmarks_numba(
            normalized_landmark_list, width, height, depth_image, intrinsic_matrix.ppx, intrinsic_matrix.ppy,
            intrinsic_matrix.fx, intrinsic_matrix.fy)

# @jit(nopython=True)
def get_center_3d_numba(face_iris_landmark, left_mm, right_mm, width, height, ppx, ppy, fx, fy):
    # import IPython; IPython.embed()

    center = np.mean(face_iris_landmark, axis=0)
    face_x_pix = center[0] * width
    face_y_pix = center[1] * height
    face_z = (left_mm + right_mm) / 2
    face_x = face_z * (face_x_pix - ppx) / fx
    face_y = face_z * (face_y_pix - ppy) / fy

    # Because y is up-axis in most cases.
    # face_y = -face_y

    return [face_x, face_y, face_z]

def get_face_center_3d(
        face_iris_landmark, left_mm, right_mm, width, height, intrinsic_matrix):
    return get_center_3d_numba(face_iris_landmark, left_mm, right_mm, width, height, \
        intrinsic_matrix.ppx, intrinsic_matrix.ppy, intrinsic_matrix.fx, intrinsic_matrix.fy)

def project_point(point, width, height, intrinsic_matrix):
    return [
        int(point[0] * intrinsic_matrix.fx / point[2] + intrinsic_matrix.ppx),
        int(point[1] * intrinsic_matrix.fy / point[2] + intrinsic_matrix.ppy)
    ]


WRIST_IDS = [0]

FINGER_IDS = dict(
    index_finger=[
        5, 6, 7, 8
    ],
    middle_finger=[
        9, 10, 11, 12
    ],
    ring_finger=[
        13, 14, 15, 16
    ],
    pinky_finger=[
        17, 18, 19, 20
    ],
    thumb_finger=[
        1, 2, 3, 4
    ]
)

def get_shortest_rotvec_between_two_vector(a, b):
    """Get shortest rotation between two vectors.

    Args:
        a - starting vector of rotation
        b - destination vector of rotation

    Returns:
        rotation_axis - axis of rotation
        theta - theta of rotation (in radian)
    """

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot = a.dot(b)
    # st.write(dot)
    if (1 - dot) < 1e-10:
        return None

    # Because they are unit vectors.
    theta = np.arccos(dot)

    rotation_axis = np.cross(a, b)

    return rotation_axis, theta

import pikapi.landmark_utils

def get_relative_angles_from_xy_plain(position_array):
    """Get relative angles from xy plain.

    Arguments:
        position_array {ndarray of shape (N,3)} -- array to get

    Returns:
        list -- Relative angles
    """
    base = position_array[-1] - position_array[0]
    base[2] = 0
    # st.write(base)

    finger_diff = position_array[1:] - position_array[:-1]
    finger_diff = np.concatenate((np.expand_dims(base, 0), finger_diff))

    # st.write(finger_diff)
    return [
        pikapi.landmark_utils.get_shortest_rotvec_between_two_vector(a, b)
        for a, b in zip(finger_diff[1:], finger_diff[:-1])
    ]


def get_relative_angles_from_finger_base(landmark, finger_ids):
    # finger_pos = landmark[WRIST_IDS + finger_ids]
    finger_pos = landmark[finger_ids]
    finger_diff = finger_pos[1:] - finger_pos[:-1]

    # rotations = []
    # for a, b in zip(finger_diff[1:], finger_diff[:-1]):
    #     rotations.append(get_shortest_rotvec_between_two_vector2(a, b))

    return [
        get_shortest_rotvec_between_two_vector(a, b)
        for a, b in zip(finger_diff[1:], finger_diff[:-1])
    ]


def get_relative_angles_to_match_to_followee(landmark, finger_ids):
    # finger_pos = landmark[WRIST_IDS + finger_ids]
    finger_pos = landmark[finger_ids]
    finger_diff = finger_pos[1:] - finger_pos[:-1]
    # print(finger_diff)

    # rotations = []
    # for a, b in zip(finger_diff[1:], finger_diff[:-1]):
    #     rotations.append(get_shortest_rotvec_between_two_vector2(a, b))

    return [
        get_shortest_rotvec_between_two_vector(a, b)
        for a, b in zip(finger_diff[:-1], finger_diff[1:])
    ]

import pikapi.landmark_utils
def get_palm_angle(hand_landmark):
    rotvecs = []
    base = np.array([0, 0, 1.0])
    HAND_PLAIN_INDICES = [5, 9, 13, 17]
    for hand1, hand2 in zip(HAND_PLAIN_INDICES[:-1], HAND_PLAIN_INDICES[1:]):
        a = hand_landmark[hand1] - hand_landmark[0]
        b = hand_landmark[hand2] - hand_landmark[0]
        # rot_axis, angle = get_shortest_rotvec_between_two_vecotr2(a, b)
        rot_axis, _ = get_shortest_rotvec_between_two_vector(a, b)
        rotvecs.append(rot_axis)

    # return Rotation.from_rotvec(rotvecs).mean().as_quat()
    mean_direction = np.mean(rotvecs, axis=0)
    # print(mean_direction)
    mean_rotation = get_shortest_rotvec_between_two_vector(base, mean_direction)
    if mean_rotation is None:
        return None

    # print(mean_rotation)
    axis, theta = mean_rotation
    return Rotation.from_rotvec(axis * theta)


def depth_from_maybe_points_3d(hand_landmark_points):
    zs = hand_landmark_points[:, 2]
    zs = zs[~np.isnan(zs)]
    zs = zs[(zs < np.percentile(zs, 90)) & (zs > np.percentile(zs, 10))]
    if len(zs) == 0:
        return None

    return np.mean(zs)


def visualize_landmark_list(landmark_list, width, height, image):
    for i, point in enumerate(landmark_list):
        cv2.circle(image, (int(point[0] * width), int(
            point[1] * height)), 3, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(image, str(i), (int(point[0] * width), int(
            point[1] * height)), cv2.FONT_HERSHEY_PLAIN, 1.0,
            (255, 255, 255), 1, cv2.LINE_AA)

@jit(nopython=True)
def get_denormalized_landmark_list(landmark_list, width, height):
    denormalized_landmark = np.copy(landmark_list)
    denormalized_landmark[:, 0] *= width
    denormalized_landmark[:, 2] *= width
    denormalized_landmark[:, 1] *= height

    return denormalized_landmark
