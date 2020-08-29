from collections import deque

from scipy.spatial.transform.rotation import Rotation
import pikapi.facelandmark_utils as flu

import numpy as np

def plot_xy(timestamps, *args):
    import plotly.graph_objects as go
    import streamlit as st
    data = []
    timestamps = [timestamp - timestamps[0] for timestamp in timestamps]
    for i, seq in enumerate(args):
        data.append(go.Scatter(x=timestamps, y=seq, name=f"data_{i}", line = dict(width=4, dash='dot')))

    fig = go.Figure(
        data=data
    )
    fig.update_layout(xaxis_title='timestamp')
    st.write(fig)

def get_projected_length(vec, base):
    # print(vec)
    # a_m = np.linalg.norm(vec)
    return vec.dot(base) / np.sqrt(np.linalg.norm(base))

# import streamlit as st

# WINDOW_PERIOD_S_ = 0.8

# Shorter period is better for gradual change.
WINDOW_PERIOD_S_ = 0.5
def get_minmax_feature_from_base_dir(face_landmark_lists, center):
    recent_data = []
    timestamps = []
    latest_timestamp = face_landmark_lists[-1].timestamp
    for face_landmark_list in face_landmark_lists:
        # if (latest_timestamp - WINDOW_PERIOD_S_) < face_landmark_list.timestamp:
        if len(face_landmark_list.data) == 0:
            continue
        main_face = face_landmark_list.data[0]
        timestamps.append(face_landmark_list.timestamp)
        recent_data.append(main_face)

    if len(recent_data) < 10:
        return None, None, None, None
    
    dir_vecs = []
    for recent_datum in recent_data:
        dir_vecs.append(flu.simple_face_direction(recent_datum, only_direction=True))

    # first_part_slice = slice(0, int(len(dir_vecs)/5))
    first_part_slice = slice(0, int(len(dir_vecs)))
    estimated_neatral_dir_vec = np.mean(dir_vecs[first_part_slice], axis=0)
    estimated_neautral_down = np.zeros((3,))
    estimated_neautral_right = np.zeros((3,))
    for recent_datum in recent_data:
        estimated_neautral_down += (
            np.array(recent_datum[164] - recent_datum[168]) + 
            np.array(recent_datum[208] - recent_datum[10]) + 
            np.array(recent_datum[152] - recent_datum[10]) + 
            np.array(recent_datum[0] - recent_datum[8]))/4.0
        estimated_neautral_right += (
            np.array(recent_datum[266] - recent_datum[36]) +
            np.array(recent_datum[423] - recent_datum[203]) +
            np.array(recent_datum[425] - recent_datum[205]) +
            np.array(recent_datum[132] - recent_datum[361])
            )/4.0
    
    # st.write(estimated_neatral_dir_vec)
    # st.write(estimated_neautral_right)
    # st.write(estimated_neautral_down)

    xs = []
    ys = []
    # st.write(estimated_center)
    for dir_vec, recent_datum_landmark in zip(dir_vecs, recent_data):
        tup = flu.get_shortest_rotvec_between_two_vector(estimated_neatral_dir_vec, (0,0,-1))
        # if tup is not None:
        rot_ax, angle = tup

        # inv_rot = Rotation.from_rotvec(rot_ax * angle).inv()
        # new_dir_vec = inv_rot.apply(dir_vec)
        # face_right_vec = np.array(recent_datum_landmark[266] - recent_datum_landmark[36])
        # face_down_vec = np.array(recent_datum_landmark[8] - recent_datum_landmark[9])
        x = get_projected_length(dir_vec, estimated_neautral_right)
        y = get_projected_length(dir_vec, estimated_neautral_down)

        # st.write(face_right_vec)
        # st.write(face_down_vec)

        # xs.append(new_dir_vec[0])
        # ys.append(new_dir_vec[1])
        xs.append(x)
        ys.append(y)
        # else:
        #     xs.append(dir_vec[0])
        #     ys.append(dir_vec[1])

    feature = max(xs), min(xs), max(ys), min(ys)
    # st.write(feature)
    dir_vecs = np.array(dir_vecs)
    # if _get_state_from_feature(feature) == 1 or _get_state_from_feature(feature) == -1:
    #     plot_xy(timestamps, xs, ys, dir_vecs[:, 0], dir_vecs[:, 1])

    return feature

    # mean_x = np.mean(xs[:len(xs)/4])
    # mean_y = np.mean(ys[:len(ys)/4])
    # return max(xs) - mean_x, min(xs) - mean_x, max(ys) - mean_y, min(ys) - mean_y

def get_minmax_feature(face_landmark_lists, center):
    recent_data = []
    latest_timestamp = face_landmark_lists[-1].timestamp
    for face_landmark_list in face_landmark_lists:
        if (latest_timestamp - WINDOW_PERIOD_S_) < face_landmark_list.timestamp:
            if len(face_landmark_list.data) == 0:
                continue
            main_face = face_landmark_list.data[0]
            recent_data.append(main_face)

    if len(recent_data) == 0:
        return None, None, None, None

    xs = []
    ys = []
    for recent_datum in recent_data:
        dir_vec = flu.simple_face_direction(recent_datum, only_direction=True)
        xs.append(dir_vec[0])
        ys.append(dir_vec[1])

    return max(xs), min(xs), max(ys), min(ys)

def _get_state_from_feature(feature):
    xmax, xmin, ymax, ymin = feature
    if xmax is None:
        return 0

    if ymax > 0.5 and ymin <= 0.0:
        return 1
    elif xmax > 0.2 and xmin < -0.2:
        return -1
    else:
        return 0

def get_state(face_landmark_lists, center=None):
    feature = get_minmax_feature_from_base_dir(face_landmark_lists, center)
    return _get_state_from_feature(feature)

class YesOrNoEstimator:
    def __init__(self):
        self.face_landmark_lists = deque([]) 
        self.initial_direction = None
    
    def get_state(self, face_landmark_list):
        if len(face_landmark_list.data) > 0 and self.initial_direction is not None:
            self.initial_direction = flu.simple_face_direction(face_landmark_list.data[0], only_direction=True)

        self.face_landmark_lists.append(face_landmark_list)
        
        # Assumption: timestamps are ascending order.
        num_remove = 0
        for elem in self.face_landmark_lists:
            if elem.timestamp < (face_landmark_list.timestamp - WINDOW_PERIOD_S_):
                num_remove += 1
            else:
                break

        for _ in range(0, num_remove):
            self.face_landmark_lists.popleft()

        # print(self.face_landmark_lists)

        # It is hard to decide with little points.
        if len(self.face_landmark_lists) <= 10:
            return 0

        return get_state(self.face_landmark_lists, center=self.initial_direction)