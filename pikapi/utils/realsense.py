"""This is the perception server that accept RGBD camera input 
and output data through ipc (currently zeromq) in the proto message format..
"""

import cv2
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
