import numpy as np


def realsense_vec_to_unity_char_vec(vec):
    return np.array([vec[0], -vec[1], -vec[2]])