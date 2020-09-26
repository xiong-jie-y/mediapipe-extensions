

import numpy as np


class IMUInfo:
    def __init__(self, acc: np.ndarray):
        self.acc = acc

class IntrinsicMatrix():
    def __init__(self, intrinsic_matrix):
        self.fx = intrinsic_matrix.fx
        self.fy = intrinsic_matrix.fy
        self.ppx = intrinsic_matrix.ppx
        self.ppy = intrinsic_matrix.ppy
