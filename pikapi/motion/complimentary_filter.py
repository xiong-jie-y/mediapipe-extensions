import math
from functools import reduce
from typing import Optional
import numpy as np

from scipy.spatial.transform.rotation import Rotation


def rpy_to_rotation(rpy):
    rot_axises = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    rots = [
        Rotation.from_rotvec(ax * angle) for ax, angle in zip(rot_axises, rpy)
        # if abs(angle) > 0.0001
    ]
    all_rot = reduce(lambda r1, r2: r2 * r1, rots, Rotation.identity())

    return all_rot


class ComplimentaryFilterOrientationEstimator:
    """Simplest complimentary filter for realsense.

    This just combine gyro and acc with constant coefficient.
    Assumued coordinate system is 
    https://github.com/IntelRealSense/librealsense/blob/master/doc/d435i.md#sensor-origin-and-coordinate-system
    """

    def __init__(self, alpha=0.98):
        self.previous_timestamp = None
        self.rotation: Optional[Rotation] = None
        self.alpha = alpha

    def process_gyro(self, gyro_vector: np.ndarray, timestamp: float):
        if self.previous_timestamp is None:
            self.previous_timestamp = timestamp
        else:
            div_t = timestamp - self.previuos_timestamp
            div_pyr = np.array([gyro_vector[0], gyro_vector[1], gyro_vector[2]])
            div_pyr = div_pyr * div_t / 1000.0
            div_rotation = rpy_to_rotation(div_pyr)
            self.rotation = div_rotation * self.rotation
            self.previous_timestamp = timestamp

    def process_accel(self, acc_vector: np.ndarray):
        """
        Use acceleration vector to estimated rotation.

        Arguments:
            - acc_vector: The direction of gravtiy.
        """
        # pitch = -math.atan2(acc_vector[2], acc_vector[1])
        pitch = math.pi / 2
        roll = -math.atan2(acc_vector[0], acc_vector[1])

        print(pitch, roll)

        self.rotation = rpy_to_rotation(np.array([roll, pitch, 0]))
        # if self.rotation is None or self.rotation.magnitude() < 0.001:
        #     self.rotation = rpy_to_rotation(np.array([pitch, 0, roll]))
        # else:
        #     self.rotation = (rpy_to_rotation((1. - self.alpha) * np.array([pitch, 0, roll]))) \
        #         * Rotation.from_rotvec(self.alpha * self.rotation.as_rotvec())

    def get_pose(self):
        return self.rotation.as_quat()
