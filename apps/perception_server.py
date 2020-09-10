"""This is the perception server that accept RGBD camera input 
and output data through ipc (currently zeromq) in the proto message format..
"""
import logging
from typing import List
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
    cap.release()

    return get_intrinsic(int(width), int(height))

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

    return np.array(points_c)


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
        for a,b in zip(finger_diff[1:], finger_diff[:-1])
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
        for a,b in zip(finger_diff[1:], finger_diff[:-1])
    ]

class HandGestureRecognizer():
    def __init__(self, intrinsic_matrix):
        self.hand_recognizer = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/multi_hand_tracking_gpu.pbtxt", [
                "gesture_texts", "multi_hand_landmarks"], {})
        self.logger = logging.getLogger(__class__.__name__)

        # TODO: Change to camera object
        self.intrinsic_matrix = intrinsic_matrix

        self.record_trajectory = False

    def _get_achimuitehoi_gesture(self, landmark_list: np.ndarray, 
        width: int, height: int, visualize_image: np.ndarray, min_x: int, min_y: int) -> str:
        denormalized_landmark = np.copy(landmark_list)
        denormalized_landmark[:,0] *= width
        denormalized_landmark[:,2] *= width
        denormalized_landmark[:,1] *= height

        max_angles = {}
        for finger_name in FINGER_IDS.keys():
            # rotations = get_relative_angles_from_xy_plain(
            #     landmark_list[FINGER_IDS[finger_name]])
            rotations = get_relative_angles_from_finger_base(landmark_list, FINGER_IDS[finger_name])
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
                direction="camera"
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
        cv2.putText(visualize_image, \
            state_tag, 
            (min_x, min_y), cv2.FONT_HERSHEY_PLAIN, 2.0,
            (0, 0, 255), 3, cv2.LINE_AA)

        pressed_key = cv2.waitKey(2)
        if pressed_key == ord("b"):
            if state_tag == "pointing_to_up":
                import IPython; IPython.embed()

        return state_tag

    def _accumulate_trajectory(self, landmark_list: np.ndarray, 
        width: int, height: int, visualize_image: np.ndarray):
        pressed_key = cv2.waitKey(2)
        if pressed_key == ord("t"):
            self.logger.info("Start Logging")
            self.trajectory = []
            self.record_trajectory = True

        if self.record_trajectory:
            self.trajectory.append(landmark_list[FINGER_IDS['index_finger']][-1])

            for first, second in zip(self.trajectory[:-1], self.trajectory[1:]):
                cv2.line(visualize_image, (int(first[0] * width), int(
                    first[1] * height)), (int(second[0] * width), int(
                    second[1] * height)), (255, 255, 255), 5)

    def get_gestures(self,
                     rgb_image: np.ndarray, depth_image: np.ndarray,
                     visualize_image: np.ndarray) -> List[str]:
        width = rgb_image.shape[1]
        height = rgb_image.shape[0]

        # Estimate hand landmarks and gesture texts
        self.hand_recognizer.process_frame(rgb_image)
        hand_landmarks = np.array(
            self.hand_recognizer.get_normalized_landmark_lists("multi_hand_landmarks"))
        gesture_texts = self.hand_recognizer.get_string_array("gesture_texts")

        effective_gesture_texts = []
        assert len(hand_landmarks) == len(gesture_texts)
        acchimuitehoi_gesture_name = None
        for hand_landmark_list, gesture_text in zip(hand_landmarks, gesture_texts):
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
            effective_gesture_texts.append(gesture_text)
            for i, point in enumerate(hand_landmark_list):
                cv2.circle(visualize_image, (int(point[0] * width), int(
                    point[1] * height)), 3, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
                cv2.putText(visualize_image, str(i), (int(point[0] * width), int(
                    point[1] * height)), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(visualize_image, gesture_text, (min_x, min_y), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (255, 255, 255), 1, cv2.LINE_AA)

            effective_gesture_texts.append(self._get_achimuitehoi_gesture(
                hand_landmark_list, width, height, visualize_image, min_x, min_y + 50))
            self._accumulate_trajectory(hand_landmark_list, width, height, visualize_image)

        return effective_gesture_texts


@click.command()
@click.option('--camera-id', '-c', default=0, type=int)
@click.option('--relative-depth', is_flag=True)
@click.option('--use-realsense', is_flag=True)
@click.option('--demo-mode', is_flag=True)
def cmd(camera_id, relative_depth, use_realsense, demo_mode):
    intrinsic_matrix = get_intrinsic_matrix(camera_id)
    focal_length = intrinsic_matrix.fx

    runner = pikapi.graph_runner.GraphRunner(
        "pikapi/graphs/iris_tracking_gpu.pbtxt", [
            "face_landmarks_with_iris", "left_iris_depth_mm", "right_iris_depth_mm"],
        pmu.create_packet_map({"focal_length_pixel": focal_length})
    )
    hand_gesture_recognizer = HandGestureRecognizer(intrinsic_matrix)
    pose_recognizer = pikapi.graph_runner.GraphRunner(
        "pikapi/graphs/upper_body_pose_tracking_gpu.pbtxt", [
            "pose_landmarks"], {})


    import zmq

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://127.0.0.1:9998")

    import pikapi.protos.perception_state_pb2 as ps

    # Alignオブジェクト生成
    config = rs.config()
    #config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    #config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    #
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 60)
    #
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 60)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    import time
    from pikapi.head_gestures import YesOrNoEstimator
    yes_or_no_estimator = YesOrNoEstimator()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)

    previous = None
    try:
        while(True):
            current_time = time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue
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
            processed_frame = runner.process_frame(gray)
            processed_frame = processed_frame.reshape(
                (frame.shape[0], frame.shape[1], 4))
            width = frame.shape[1]
            height = frame.shape[0]
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

            # multi_face_landmarks = runner.get_normalized_landmark_lists(
            #     "multi_face_landmarks")

            # print(processed_frame.shape)
            blank_image = np.zeros(processed_frame.shape, np.uint8)
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

            multi_face_landmarks = [runner.get_normalized_landmark_list(
                "face_landmarks_with_iris")]
            left_mm = runner.get_float("left_iris_depth_mm")
            right_mm = runner.get_float("right_iris_depth_mm")
            if left_mm is not None and right_mm is not None:
                # cv2.putText(blank_image, str(int((left_mm + right_mm)/2)) + "[mm]", (0, 50), cv2.FONT_HERSHEY_PLAIN, 4.0,
                #     (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(target_image, f"Left: {int(left_mm)} [mm]", (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(target_image, f"Right: {int(right_mm)} [mm]", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2.0,
                            (255, 255, 255), 2, cv2.LINE_AA)

            effective_gesture_texts = \
                hand_gesture_recognizer.get_gestures(gray, depth_image, target_image)

            pose_recognizer.process_frame(gray)
            pose_landmark_list = np.array(
                pose_recognizer.get_normalized_landmark_list("pose_landmarks"))

            # print(pose_landmark_list)

            if len(pose_landmark_list) > 0:
                # if not is_too_far(pose_landmark_list, width, height, depth_image):
                # mean_depth = get_mean_depth(pose_landmark_list, width, height, depth_image)
                # if np.mean(mean_depth) <= 1500:
                for point in pose_landmark_list:
                    cv2.circle(target_image, (int(point[0] * frame.shape[1]), int(
                        point[1] * frame.shape[0])), 3, (255, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
                # else:
                #     print(mean_depth)
            center = None
            if len(multi_face_landmarks) != 0 and (
                left_mm is not None and right_mm is not None
            ):
                face_landmark = np.array(multi_face_landmarks[0])
                from pikapi.logging import TimestampedData
                if len(face_landmark) != 0:
                    min_x = int(min(face_landmark[:, 0]) * frame.shape[1])
                    min_y = int(min(face_landmark[:, 1]) * frame.shape[0])
                    state = yes_or_no_estimator.get_state(
                        TimestampedData(current_time, multi_face_landmarks))
                    if state == 1:
                        state = "Yes"
                    elif state == -1:
                        state = "No"
                    else:
                        state = "No Gesture"

                    cv2.putText(target_image, state, (min_x, min_y), cv2.FONT_HERSHEY_PLAIN, 1.0,
                                (255, 255, 255), 1, cv2.LINE_AA)

                    # print(face_mesh_to_camera_coordinate(face_landmark, left_mm, right_mm))
                    center = get_face_center_3d(
                        face_landmark, left_mm, right_mm, width, height, intrinsic_matrix)
                    # This center will have a y-axis down side.
                    x, y = project_point(
                        center, width, height, intrinsic_matrix)
                    # print(center)
                    # print(x,y)

                    for point in face_landmark:
                        cv2.circle(target_image, (int(point[0] * frame.shape[1]), int(
                            point[1] * frame.shape[0])), 3, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(target_image, (x, y), 3, (0, 0, 255),
                               thickness=-1, lineType=cv2.LINE_AA)
                    # print(np.mean([point[2] for point in face_landmark]))

            if not demo_mode:
                target_image = np.hstack((target_image, depth_colored))

            cv2.imshow("Frame", target_image)

            pressed_key = cv2.waitKey(3)
            if pressed_key == ord("a"):
                break

            # def array_to_vec(array, previous):
            if center is None:
                vec = previous
            else:
                vec = ps.Vector(x=center[0], y=-center[1], z=center[2])
                previous = vec

            # print(effective_gesture_texts)
            perc_state = ps.PerceptionState(
                people=[
                    ps.Person(
                        face=ps.Face(center=vec),
                        hands=[
                            ps.Hand(
                                # gesture_type=ps.Hand.GestureType.WAVING \
                                #     if 'FOUR' in effective_gesture_texts or 'FIVE' in effective_gesture_texts \
                                #         else ps.Hand.GestureType.NONE
                                gesture_names=effective_gesture_texts
                            )
                        ]
                    )
                ]
            )

            data = ["PerceptionState".encode(
                'utf-8'), perc_state.SerializeToString()]
            publisher.send_multipart(data)

    finally:
        # ストリーミング停止
        pipeline.stop()


def main():
    cmd()


if __name__ == "__main__":
    main()
