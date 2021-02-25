from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

@dataclass
class Timestamped():
    timestamp: float

@dataclass
class MotionFrame(Timestamped):
    acc: np.array = None
    gyro: np.array = None

@dataclass
class ImageFrame(Timestamped):
    data: np.array = None

@dataclass
class IMUInfo:
    acc: np.array = None