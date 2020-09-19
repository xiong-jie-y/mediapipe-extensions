

import numpy as np


class IMUInfo:
    def __init__(self, acc: np.ndarray):
        self.acc = acc