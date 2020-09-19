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


def get_intrinsic_matrix(camera_id):
    cap = cv2.VideoCapture(camera_id)
    width = cap.get(3)
    height = cap.get(4)
    print(width, height)
    cap.release()

    # return get_intrinsic(int(width), int(height))
    return get_intrinsic(int(width), 360)

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


def get_camera_coord_landmarks(
        normalized_landmark_list: np.ndarray, width: int, height: int,
        depth_image: np.ndarray, intrinsic_matrix: np.ndarray):
    """Get landmark list in camera coordinate.

    Returns:
        The 3d points in camera coord in [x,y,z].
    """
    points_c = []
    for point in normalized_landmark_list:
        # These points are outside of the depth images.
        if point[0] < 0.0 or point[0] > 1.0 or point[1] < 0.0 or point[1] > 1.0:
            points_c.append([point[0], point[1], None])
            continue

        x_pix = int(point[1] * height)
        y_pix = int(point[0] * width)
        depth_value = depth_image[x_pix, y_pix]

        # This depth value is invalid.
        if depth_value == 0:
            points_c.append([point[0], point[1], None])
            continue

        # Get 3d coordinate from depth.
        x = depth_value * (x_pix - intrinsic_matrix.ppx) / intrinsic_matrix.fx
        y = depth_value * (y_pix - intrinsic_matrix.ppy) / intrinsic_matrix.fy
        points_c.append([x, y, depth_value])

    assert len(points_c) == len(normalized_landmark_list)

    return np.array(points_c, dtype=np.float)


def get_face_center_3d(
        face_iris_landmark, left_mm, right_mm, width, height, intrinsic_matrix):
    # import IPython; IPython.embed()

    center = np.mean(face_iris_landmark, axis=0)
    face_x_pix = center[0] * width
    face_y_pix = center[1] * height
    face_z = (left_mm + right_mm) / 2
    face_x = face_z * (face_x_pix - intrinsic_matrix.ppx) / intrinsic_matrix.fx
    face_y = face_z * (face_y_pix - intrinsic_matrix.ppy) / intrinsic_matrix.fy

    # Because y is up-axis in most cases.
    # face_y = -face_y

    return [face_x, face_y, face_z]


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
        get_shortest_rotvec_between_two_vector(a, b)
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
    print(finger_diff)
    
    # rotations = []
    # for a, b in zip(finger_diff[1:], finger_diff[:-1]):
    #     rotations.append(get_shortest_rotvec_between_two_vector2(a, b))

    return [
        get_shortest_rotvec_between_two_vector(a, b)
        for a, b in zip(finger_diff[:-1], finger_diff[1:])
    ]

def get_palm_angle(hand_landmark):
    rotvecs = []
    base = np.array([0, 0, 1])
    HAND_PLAIN_INDICES = [5, 9, 13, 17]
    for hand1, hand2 in zip(HAND_PLAIN_INDICES[:-1], HAND_PLAIN_INDICES[1:]):
        a = hand_landmark[hand1] - hand_landmark[0] 
        b = hand_landmark[hand2] - hand_landmark[0] 
        # rot_axis, angle = get_shortest_rotvec_between_two_vector2(a, b)
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


def get_denormalized_landmark_list(landmark_list, width, height):
    denormalized_landmark = np.copy(landmark_list)
    denormalized_landmark[:, 0] *= width
    denormalized_landmark[:, 2] *= width
    denormalized_landmark[:, 1] *= height

    return denormalized_landmark


class HandGestureRecognizer():
    def __init__(self, intrinsic_matrix):
        self.hand_recognizer = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/multi_hand_tracking_gpu.pbtxt", [
                "gesture_texts", "multi_hand_landmarks", "multi_handedness"], {})
        self.logger = logging.getLogger(__class__.__name__)

        # TODO: Change to camera object
        self.intrinsic_matrix = intrinsic_matrix

        self.record_trajectory = False

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

    def _get_fingers(self, landmark_list, width, height) -> List[ps.Finger]:
        denormalized_landmark_list = get_denormalized_landmark_list(
            landmark_list, width, height) 
        palm_rotation = get_palm_angle(denormalized_landmark_list)

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
                ps.Finger(finger_name=finger_name, rotations=[
                    self._proto_quaternion_from_rotation(
                        Rotation.from_rotvec(realsense_vec_to_unity_char_vec(axis) * theta)) for axis, theta in rotations
                ], rotation_angles = [
                    theta for axis, theta in rotations
                ]
                )
            )

        return finger_states

    def _accumulate_trajectory(self, landmark_list: np.ndarray,
                               width: int, height: int, visualize_image: np.ndarray):
        pressed_key = cv2.waitKey(2)
        if pressed_key == ord("t"):
            self.logger.info("Start Logging")
            self.trajectory = []
            self.record_trajectory = True

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

        # Estimate hand landmarks and gesture texts
        self.hand_recognizer.process_frame(rgb_image)
        hand_landmarks = np.array(
            self.hand_recognizer.get_normalized_landmark_lists("multi_hand_landmarks"))
        gesture_texts = self.hand_recognizer.get_string_array("gesture_texts")
        from mediapipe.framework.formats.classification_pb2 import ClassificationList
        multi_handedness = self.hand_recognizer.get_proto_list(
            "multi_handedness")
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

        hand_states = []
        assert len(hand_landmarks) == len(gesture_texts)
        acchimuitehoi_gesture_name = None
        for hand_landmark_list, gesture_text, handedness in zip(hand_landmarks, gesture_texts, multi_handedness_parsed):
            effective_gesture_texts = []
            hand_landmark_points = get_camera_coord_landmarks(
                hand_landmark_list, width, height, depth_image, self.intrinsic_matrix)

            # Filter conversing person's hand.
            zs = hand_landmark_points[:, 2]
            zs = zs[zs != None]
            if len(zs) == 0:
                continue
            mean_depth = np.mean(zs)

            if mean_depth > 1500:
                for i, point in enumerate(hand_landmark_list):
                    cv2.circle(visualize_image, (int(point[0] * width), int(
                        point[1] * height)), 3, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                continue
            # self.logger.info(f"In a range: {mean_depth}")

            min_x = int(min(hand_landmark_list[:, 0]) * width)
            min_y = int(min(hand_landmark_list[:, 1]) * height)
            max_x = int(max(hand_landmark_list[:, 0]) * width)
            max_y = int(max(hand_landmark_list[:, 1]) * height)
            import pikapi.utils.opencv
            pikapi.utils.opencv.overlay_rect_with_opacity(
                visualize_image, (min_x, min_y, (max_x - min_x), (max_y - min_y)))
            effective_gesture_texts.append(gesture_text)
            visualize_landmark_list(
                hand_landmark_list, width, height, visualize_image)

            cv2.putText(visualize_image, gesture_text, (min_x, min_y + 25), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 0, 0), 2, cv2.LINE_AA)
            # print(handedness.classification[0].label)
            cv2.putText(visualize_image, handedness.classification[0].label, (min_x, min_y + 100), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 0, 0), 2, cv2.LINE_AA)

            effective_gesture_texts.append(self._get_achimuitehoi_gesture(
                hand_landmark_list, width, height, visualize_image, min_x, min_y + 50))
            self._accumulate_trajectory(
                hand_landmark_list, width, height, visualize_image)

            hand_states.append(ps.Hand(
                gesture_names=effective_gesture_texts,
                hand_exist_side=handedness.classification[0].label,
                fingers=self._get_fingers(
                    hand_landmark_list, width, height)
            ))

        # print(hand_states)
        return hand_states


class IMUInfo:
    def __init__(self, acc: np.ndarray):
        self.acc = acc


NOSE_INDEX = 4


class FaceRecognizer:
    def __init__(self, intrinsic_matrix):
        focal_length = intrinsic_matrix.fx
        self.runner = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/iris_tracking_gpu.pbtxt", [
                "face_landmarks_with_iris", "left_iris_depth_mm", "right_iris_depth_mm"],
            pmu.create_packet_map({"focal_length_pixel": focal_length})
        )
        self.intrinsic_matrix = intrinsic_matrix
        self.yes_or_no_estimator = YesOrNoEstimator()
        self.previous = None

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
        denormalized_landmark = get_denormalized_landmark_list(
            landmark_list, width, height)
        center = np.mean(denormalized_landmark, axis=0)
        center_to_nose_direction = denormalized_landmark[NOSE_INDEX] - center
        # print(center_to_nose_direction)
        up_direction = denormalized_landmark[9] - denormalized_landmark[164]
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

        def camera_coord_to_unity_coord(array):
            return np.array([array[0], -array[1], -array[2]])

        # Transform pose axis to unity world coordinate.
        up = camera_coord_to_unity_coord(up_direction / np.linalg.norm(up_direction))
        front = camera_coord_to_unity_coord(center_to_nose_direction / np.linalg.norm(center_to_nose_direction))
        right = np.cross(up, front)
        right = right / np.linalg.norm(right)

        rot = Rotation.align_vectors([right, up, front], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        rot_vec = rot[0].as_rotvec()
        debug_txt = f"""Rotation
        {rot_vec}
        {np.linalg.norm(rot_vec)}
        {rot[1]}
        """

        cv2.putText(face_image, debug_txt, (0, 50), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # return np.array(
        #     [-center_to_nose_direction[0],
        #      -center_to_nose_direction[1],
        #      -center_to_nose_direction[2]]
        # )
        return rot_vec

    def get_face_state(self,
                       rgb_image: np.ndarray, depth_image: np.ndarray,
                       visualize_image: np.ndarray, imu_info: IMUInfo) -> ps.Face:
        current_time = time.time()
        width = rgb_image.shape[1]
        height = rgb_image.shape[0]

        self.runner.process_frame(rgb_image)

        multi_face_landmarks = [self.runner.get_normalized_landmark_list(
            "face_landmarks_with_iris")]
        left_mm = self.runner.get_float("left_iris_depth_mm")
        right_mm = self.runner.get_float("right_iris_depth_mm")
        # if left_mm is not None and right_mm is not None:
        #     # cv2.putText(blank_image, str(int((left_mm + right_mm)/2)) + "[mm]", (0, 50), cv2.FONT_HERSHEY_PLAIN, 4.0,
        #     #     (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.putText(visualize_image, f"Left: {int(left_mm)} [mm]", (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0,
        #                 (255, 255, 255), 2, cv2.LINE_AA)
        #     cv2.putText(visualize_image, f"Right: {int(right_mm)} [mm]", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2.0,
        #                 (255, 255, 255), 1, cv2.LINE_AA)

        center = None
        center_to_nose_direction = None
        if len(multi_face_landmarks) != 0 and (
            left_mm is not None and right_mm is not None
        ):
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

                direction = self._get_face_direction(
                    face_landmark, width, height, visualize_image, face_image)
                # direction = self._adjust_by_imu(
                #     np.array([direction]), imu_info)[0]
                center_to_nose_direction = ps.Vector(
                    x=direction[0], y=direction[1], z=direction[2])

                cv2.imshow("Face Image", face_image)

                cv2.circle(visualize_image, (x, y), 3, (0, 0, 255),
                           thickness=-1, lineType=cv2.LINE_AA)
                # print(np.mean([point[2] for point in face_landmark]))

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

        # print("Face Direction")
        # print(center_to_nose_direction)
        return ps.Face(
            center=vec, eye_camera_position=eye_camera_position, eye_camera_pose=camera_pose,
            character_pose=character_pose,
            center_in_unity=center_in_unity_pb,
            direction_vector_in_unity=center_to_nose_direction
        )


class BodyRecognizer():
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
    
        print(denormalized_landmark_list[15, :])

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
            pressed_key = cv2.waitKey(2)
            if pressed_key == ord("b"):
                import IPython
                IPython.embed()
            bones.append(ps.Bone(pose=quat, name=name, z_angle=-theta))

        # print(bones)
        return bones

    def get_body_state(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                       visualize_image: np.ndarray):
        width = rgb_image.shape[1]
        height = rgb_image.shape[0]

        self.pose_recognizer.process_frame(rgb_image)
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


@click.command()
@click.option('--camera-id', '-c', default=0, type=int)
@click.option('--relative-depth', is_flag=True)
@click.option('--use-realsense', is_flag=True)
@click.option('--demo-mode', is_flag=True)
def cmd(camera_id, relative_depth, use_realsense, demo_mode):
    intrinsic_matrix = get_intrinsic_matrix(camera_id)

    face_recognizer = FaceRecognizer(intrinsic_matrix)
    hand_gesture_recognizer = HandGestureRecognizer(intrinsic_matrix)
    body_recognizer = BodyRecognizer(intrinsic_matrix)
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
            print(estimated_fps)
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
            data = ["PerceptionState".encode(
                'utf-8'), perc_state.SerializeToString()]
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
