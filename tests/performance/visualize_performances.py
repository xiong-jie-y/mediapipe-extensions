import time

import sys;sys.path.append(".")

import numpy as np

vectors1 = np.random.rand(1000, 3)
vectors2 = np.random.rand(1000, 3)

from pikapi.utils.development import compare_function_performances

import pikapi.utils.landmark
import pikapi.landmark_utils

compare_function_performances([
    pikapi.landmark_utils.get_shortest_rotvec_between_two_vector,
    pikapi.utils.landmark.get_shortest_rotvec_between_two_vector,
    ],
    list(zip(vectors1, vectors2))
)

vectors1 = np.random.rand(1000, 21, 3)
finger_ids = [pikapi.utils.landmark.FINGER_IDS] * 1000

compare_function_performances([
    pikapi.landmark_utils.get_fingers
    ],
    list(zip(vectors1, finger_ids))
)