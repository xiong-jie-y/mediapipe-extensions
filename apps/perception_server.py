# This is an example to run the graph in python.
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

@click.command()
@click.option('--camera-id', '-c', default=0, type=int)
@click.option('--relative-depth', is_flag=True)
def cmd(camera_id, relative_depth):
    intrinsic_matrix = get_intrinsic_matrix(camera_id)
    focal_length = intrinsic_matrix.fx

    cap = cv2.VideoCapture(camera_id)

    if not relative_depth:
        runner = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/iris_tracking_gpu.pbtxt", [
                "face_landmarks_with_iris", "left_iris_depth_mm", "right_iris_depth_mm"],
            pmu.create_packet_map({"focal_length_pixel": focal_length})
        )
    else:
        runner = pikapi.graph_runner.GraphRunner(
            "pikapi/graphs/face_mesh_desktop_live_any_model.pbtxt", [
                "multi_face_landmarks"],
            pmu.create_packet_map({
                "detection_model_file_path": pikapi.get_data_path("models/face_detection_front.tflite"),
                "landmark_model_file_path": pikapi.get_data_path("models/face_landmark.tflite"),
            })
        )

    import zmq

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://127.0.0.1:9998")

    import pikapi.protos.perception_state_pb2 as ps

    previous = None
    while(True):
        ret, frame = cap.read()
        if frame is None:
            break

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

        multi_face_landmarks = [runner.get_normalized_landmark_list(
            "face_landmarks_with_iris")]
        left_mm = runner.get_float("left_iris_depth_mm")
        right_mm = runner.get_float("right_iris_depth_mm")
        if left_mm is not None and right_mm is not None:
            # cv2.putText(blank_image, str(int((left_mm + right_mm)/2)) + "[mm]", (0, 50), cv2.FONT_HERSHEY_PLAIN, 4.0,
            #     (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(blank_image, f"Left: {left_mm} [mm]", (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(blank_image, f"Right: {right_mm} [mm]", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2.0,
                        (255, 255, 255), 1, cv2.LINE_AA)


        center = None
        if len(multi_face_landmarks) != 0 and (
            left_mm is not None and right_mm is not None
        ):
            face_landmark = multi_face_landmarks[0]
            if len(face_landmark) != 0:
                # print(face_mesh_to_camera_coordinate(face_landmark, left_mm, right_mm))
                center = get_face_center_3d(
                    face_landmark, left_mm, right_mm, width, height, intrinsic_matrix)
                # This center will have a y-axis down side.
                x, y = project_point(center, width, height, intrinsic_matrix)
                # print(center)
                # print(x,y)
    
                for point in face_landmark:
                    cv2.circle(blank_image, (int(point[0] * frame.shape[1]), int(
                        point[1] * frame.shape[0])), 3, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(blank_image, (x, y), 3, (0,0, 255), thickness=-1, lineType=cv2.LINE_AA)
                # print(np.mean([point[2] for point in face_landmark]))

        cv2.imshow("Frame", blank_image)

        pressed_key = cv2.waitKey(3)
        if pressed_key == ord("a"):
            break

        # def array_to_vec(array, previous):
        if center is None:
            vec = previous
        else:
            vec = ps.Vector(x=center[0], y=-center[1], z=center[2])
            previous = vec

        perc_state = ps.PerceptionState(
            people=[
                ps.Person(face=ps.Face(center=vec))
            ]
        )
        
        data = ["PerceptionState".encode('utf-8'), perc_state.SerializeToString()]
        publisher.send_multipart(data)



def main():
    cmd()


if __name__ == "__main__":
    main()
