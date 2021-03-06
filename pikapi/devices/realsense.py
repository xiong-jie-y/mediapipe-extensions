import pyrealsense2 as rs

class RealsenseD435i():
    pass

class RGBDWithIMU():
    def __init__(self):
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 60)
        #
        config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 60)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_data(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        acc = frames[2].as_motion_frame().get_motion_data()
        gyro = frames[3].as_motion_frame().get_motion_data()
        timestamp = frames[3].as_motion_frame().get_timestamp()

        return color_frame, depth_frame, acc, gyro, timestamp
