from multiprocessing.managers import BaseManager
import ctypes
import multiprocessing
import multiprocessing.sharedctypes
from multiprocessing.sharedctypes import Value, Array
import cv2

import numpy as np


def _visualize_core(target_image, end_flag):
    cv2.imshow("Frame", target_image)

    pressed_key = cv2.waitKey(2)
    if pressed_key == ord("a"):
        # import IPython; IPython.embed()
        end_flag.value = True
        return True
    return False

def _visualize_image(image_buf, depth_image, width, height, buf_ready, end_flag):
    target_image = np.empty((height, width, 3), dtype=np.uint8)
    depth_image_viz = np.empty((height, width), dtype=np.uint16)
    while True:
        buf_ready.wait()
        target_image = np.frombuffer(image_buf, dtype=np.uint8).reshape((height, width, 3))
        depth_image_viz = np.frombuffer(depth_image, dtype=np.uint16).reshape((height, width))
        buf_ready.clear()

        # target_image = np.hstack((target_image, depth_colored))

        # Prepare depth image for dispaly.
        depth_image_cp = np.copy(depth_image_viz)
        depth_image_cp[depth_image_cp > 2500] = 0
        depth_image_cp[depth_image_cp == 0] = 0
        depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image_cp, alpha=0.08), cv2.COLORMAP_JET)

        if _visualize_core(target_image, end_flag):
            return


class VisualizeServer():
    def __init__(self, width, height):
        self.buf1 = multiprocessing.sharedctypes.RawArray('B', height * width*3)

    # def pass_image(self, rgb_image, depth_image):
    #     self.rgb_image = rgb_image
    #     self.depth_iamge = depth_image
    def get_shared_memory(self):
        return self.buf1


class VisualizeGUI():
    def __init__(self, width: int, height: int, run_multiprocess: bool = True, demo_mode: bool = True):
        self.end_flag = Value(ctypes.c_bool, False)

        if run_multiprocess:
            self.buf1 = multiprocessing.sharedctypes.RawArray('B', height * width*3)
            self.depth_image = multiprocessing.sharedctypes.RawArray(ctypes.c_uint16, height * width)
            self.buf_ready = multiprocessing.Event()
            self.buf_ready.clear()
            self.new_image_ready_event = multiprocessing.Event()
            self.new_image_ready_event.clear()
            self.p1 = multiprocessing.Process(
                target=_visualize_image,
                args=(
                    self.buf1, self.depth_image, width, height, self.buf_ready, self.end_flag), 
                daemon=True)
            self.p1.start()

        self.run_multiprocess = run_multiprocess
        self.target_image = np.empty((height, width, 3), dtype=np.uint8)

    def pass_image(self, image, depth_image):
        if self.run_multiprocess:
            self.buf_ready.clear()
            self.new_image_ready_event.clear()
            memoryview(self.buf1).cast('B')[:] = memoryview(image).cast('B')[:]
            memoryview(self.depth_image).cast('B')[:] = memoryview(depth_image).cast('B')[:]
            self.new_image_ready_event.set()
            # buf_dict["buf"] = target_image
            # memoryview(buf1).cast('B')[:] = target_image

        # else:
        #     _visualize_core(image, self.end_flag)

    def get_image_buf(self):
        # self.target_image = np.frombuffer(self.buf1, dtype=np.uint8).view().reshape((360, 640, 3))
        bb = np.frombuffer(memoryview(self.buf1), dtype=np.uint8)
        aa = bb.reshape((360, 640, 3))
        assert np.may_share_memory(aa, bb)
        return aa

    def get_image_buf_internal(self):
        return self.buf1

    def get_depth_image_buf_internal(self):
        return self.depth_image

    def get_depth_image_buf(self):
        # self.target_image = np.frombuffer(self.buf1, dtype=np.uint8).view().reshape((360, 640, 3))
        bb = np.frombuffer(memoryview(self.depth_image), dtype=np.uint16)
        aa = bb.reshape((360, 640))
        assert np.may_share_memory(aa, bb)
        return aa

    def draw(self):
        self.buf_ready.set()

    def visualize_image(self, image):
        if self.run_multiprocess:
            self.buf_ready.clear()
            memoryview(self.buf1).cast('B')[:] = memoryview(image).cast('B')[:]
            # buf_dict["buf"] = target_image
            # memoryview(buf1).cast('B')[:] = target_image
            self.buf_ready.set()
        else:
            _visualize_core(image, self.end_flag)

    # def visualize_landmark(self, denormalized_landmark, with_index=):
    #     # self._task_queue.put(["visualize_landmark", denormalized_landmark])
    #     for i, point in enumerate(face_landmark):
    #         draw_x = int((point[0] - min_x/width) * rate * width)
    #         draw_y = int((point[1] - min_y/height) * rate * height)
    #         cv2.circle(face_image, (draw_x, draw_y), 3,
    #                 (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    #         cv2.putText(face_image, str(i), (draw_x, draw_y), cv2.FONT_HERSHEY_PLAIN, 1.0,
    #                     (255, 255, 255), 1, cv2.LINE_AA)

    def end_issued(self):
        return self.end_flag.value


def create_visualize_gui_manager() -> BaseManager:
    class MyManager(BaseManager):
        pass
    MyManager.register('VisualizeGUI', VisualizeGUI)
    MyManager.register('VisServer', VisualizeServer)
    manager = MyManager()
    manager.start()
    return manager
