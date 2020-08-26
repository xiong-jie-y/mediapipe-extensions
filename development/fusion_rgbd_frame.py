import pyrealsense2 as rs
import numpy as np
import cv2

# ストリーム(Depth/Color)の設定
config = rs.config()
#config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#
config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
#
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:

        # フレーム待ち(Color & Depth)
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        #imageをnumpy arrayに
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())


        #depth imageをカラーマップに変換
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

        #画像表示
        color_image_s = cv2.resize(color_image, (640, 360))
        depth_colormap_s = cv2.resize(depth_colormap, (640, 360))
        images = np.hstack((color_image_s, depth_colormap_s))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        if cv2.waitKey(1) & 0xff == 27:#ESCで終了
            import IPython; IPython.embed()
            
            cv2.destroyAllWindows()
            break

finally:

    #ストリーミング停止
    pipeline.stop()