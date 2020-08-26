import os
import numpy as np
import streamlit as st
import pika.logging

import pika.facelandmark_utils as flu
from pika.head_gestures import YesOrNoEstimator, get_minmax_feature_from_base_dir, get_state

def detection_analysis_dashboard():
    # @st.cache(allow_output_mutation=True)
    def load_data(result_folder):
        return pika.logging.HumanReadableLog.load_from_path(result_folder)

    def get_from_box(box):
        return ((box[2] + box[0])/2 - 250)/50 * 0.05

    def get_x_from_box(box):
        return ((box[3] + box[1])/2 - 425)

    def get_positions(detection_lists):
        positions = []
        for detection_list in detection_lists:
            found = False
            for detection in detection_list[1]:
                if detection["label"] == "onahole":
                    found = True
                    positions.append(get_x_from_box(detection["box"]))
                    break
                    
            if not found:
                if len(positions) == 0:
                    positions.append(None)
                else:
                    positions.append(positions[-1])

        return positions

    def get_sequence_from_label(sequence, labels):
        assert(len(labels) > 0)
        previous_label = labels[0].data
        partial_seq = []
        # st.write(labels)
        for elm, label in zip(sequence, labels):
            # st.write(label.data)
            if label.data == previous_label:
                partial_seq.append(elm)
            else:
                yield partial_seq, previous_label
                partial_seq = []
            previous_label = label.data

    # Choose result dir.
    result_folders_dir = st.text_input("Location of result folders.", "data/")
    result_folders = sorted(list(next(os.walk(result_folders_dir))[1]), reverse=True)
    result_folder = st.selectbox("Result folder", result_folders)

    logged_data = load_data(os.path.join("data", result_folder))

    images = logged_data.get_images_with_timestamp('input_frame')
    multi_face_landmark_lists = logged_data.get_data("multi_face_landmarks")
    motion_labels = logged_data.get_data("motion_label")

    label_to_values = {
        "Nodding": 1,
        "Shaking": -1,
    }
    label_values = [label_to_values[motion_label.data] if motion_label.data is not None else 0 for motion_label in motion_labels]
    assert(len(images) == len(multi_face_landmark_lists))

    frame_index = st.slider("Index", 0, len(images))

    st.image(images[frame_index].data)
    st.write(multi_face_landmark_lists[frame_index].data)
    st.write(flu.simple_face_direction(multi_face_landmark_lists[frame_index].data[0], only_direction=True))
    direction_ys = [
        flu.simple_face_direction(multi_face_landmarks.data[0], only_direction=True)[1] \
            if len(multi_face_landmarks.data) > 0 else None
        for multi_face_landmarks in multi_face_landmark_lists
    ]
    direction_xs = [
        flu.simple_face_direction(multi_face_landmarks.data[0], only_direction=True)[0] \
            if len(multi_face_landmarks.data) > 0 else None
        for multi_face_landmarks in multi_face_landmark_lists
    ]
    st.write(direction_ys)
    timestamps = [
        multi_face_landmarks.timestamp - multi_face_landmark_lists[0].timestamp
        for multi_face_landmarks in multi_face_landmark_lists
    ]
    # aaa = range(0, )
    # predictions = [
    #     get_state(multi_face_landmark_lists[:i])
    #     for i in range(1, len(multi_face_landmark_lists))
    # ]
    estimator = YesOrNoEstimator()
    predictions = []
    for multi_face_landmark_list in multi_face_landmark_lists:
        predictions.append(estimator.get_state(multi_face_landmark_list))

    assert len(timestamps) == len(motion_labels)

    import plotly.graph_objects as go
    fig = go.Figure(
        data=[
            go.Scatter(x=timestamps, y=direction_ys, name="y", line = dict(width=4, dash='dot')),
            go.Scatter(x=timestamps, y=direction_xs, name="x", line = dict(width=4, dash='dot')),
            go.Scatter(x=timestamps, y=label_values, name="true", line = dict(width=2)),
            go.Scatter(x=timestamps, y=predictions, name="preds", line = dict(width=4, dash='dot')),
        ],
    )
    fig.update_layout(xaxis_title='timestamp')
    st.write(fig)

    # for landmark_partial_seq, label in get_sequence_from_label(multi_face_landmark_lists, motion_labels):
    #     if label == "Nodding" or label == "Shaking":
    #         st.write(label)
    #         st.write(get_minmax_feature_from_base_dir(landmark_partial_seq, None))


    def draw_landmark_2d_with_index(landmark, filter_ids=[]):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        for i, keypoint in enumerate(landmark):
            if len(filter_ids) == 0 or i in filter_ids:
                ax1.text(keypoint[0], keypoint[1], s=str(i))

        # direction = flu.simple_face_direction(landmark)
        # direction = direction / np.linalg.norm(direction)
        # ax1.plot(direction[:,0], direction[:,1], direction[:,2])
        if len(filter_ids) == 0:
            ax1.scatter(landmark[:,0],landmark[:,1])
        else:
            ax1.scatter(landmark[filter_ids,0],landmark[filter_ids,1])
        # st.write(fig)
        plt.show()

    st.plotly_chart(get_landmark_3d_with_index_v2(np.array(multi_face_landmark_lists[frame_index].data[0])))


def get_landmark_3d_with_index_v2(landmark):
    import plotly.graph_objects as go
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=landmark[:,0],
        y=landmark[:,1],
        z=landmark[:,2],
        mode='markers'
    ))
    
    fig.update_layout(
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(
                x=1,
                y=1,
                z=1
            ),
            annotations=[
                dict(
                    showarrow=False,
                    xanchor="left",
                    xshift=10,
                    x=x,
                    y=y,
                    z=z,
                    text=str(index)
                ) for index, (x,y,z) in enumerate(landmark)
            ]
        )
    )
    return fig

detection_analysis_dashboard()
