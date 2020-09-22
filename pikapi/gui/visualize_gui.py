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
        return


def _visualize_image(image_buf, width, height, buf_ready, end_flag):
    target_image = np.empty((height, width, 3), dtype=np.uint8)
    while True:
        buf_ready.wait()
        target_image[:, :, :] = np.reshape(image_buf, (height, width, 3))
        buf_ready.clear()

        _visualize_core(target_image, end_flag)


class VisualizeGUI():
    def __init__(self, width: int, height: int, run_multiprocess: bool = True):
        self.end_flag = Value(ctypes.c_bool, False)

        if run_multiprocess:
            self.buf1 = multiprocessing.sharedctypes.RawArray('B', height * width*3)
            self.buf_ready = multiprocessing.Event()
            self.buf_ready.clear()
            self.p1 = multiprocessing.Process(target=_visualize_image,
                                              args=(self.buf1, width, height, self.buf_ready, self.end_flag), daemon=True)
            self.p1.start()

        self.run_multiprocess = run_multiprocess

    def visualize_image(self, image):
        if self.run_multiprocess:
            self.buf_ready.clear()
            memoryview(self.buf1).cast('B')[:] = memoryview(image).cast('B')[:]
            # buf_dict["buf"] = target_image
            # memoryview(buf1).cast('B')[:] = target_image
            self.buf_ready.set()
        else:
            _visualize_core(image, self.end_flag)

    # def visualize_landmark(self, denormalized_landmark, with_index=True):
    #     # self._task_queue.put(["visualize_landmark", denormalized_landmark])
    #     for i, point in enumerate(denormalized_landmark):
    #         draw_x = int((point[0] - min_x/width) * rate * width)
    #         draw_y = int((point[1] - min_y/height) * rate * height)
    #         cv2.circle(face_image, (draw_x, draw_y), 3,
    #                 (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    #         cv2.putText(face_image, str(i), (draw_x, draw_y), cv2.FONT_HERSHEY_PLAIN, 1.0,
    #                     (255, 255, 255), 1, cv2.LINE_AA)


    def end_issued(self):
        return self.end_flag.value

from multiprocessing.managers import BaseManager

def create_visualize_gui_manager() -> BaseManager:
    class MyManager(BaseManager): pass
    MyManager.register('VisualizeGUI', VisualizeGUI)
    manager = MyManager()
    manager.start()
    return manager
