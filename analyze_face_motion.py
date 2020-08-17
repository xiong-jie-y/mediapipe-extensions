import os
import streamlit as st
import result
import facelandmark_utils as flu

def detection_analysis_dashboard():
    # @st.cache(allow_output_mutation=True)
    def load_data(result_folder):
        return result.HumanReadableLog.load_from_path(result_folder)

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

    WINDOW_PERIOD_S_ = 1.0
    def get_minmax_feature(face_landmark_lists):
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

    def get_state(face_landmark_lists):
        xmax, xmin, ymax, ymin = get_minmax_feature(face_landmark_lists)
        if xmax is None:
            return 0

        if ymax > 0.4 and ymin <= 0.0:
            return 1
        elif xmax > 0.4 and xmin < -0.4:
            return -1
        else:
            return 0

    # Choose result dir.
    result_folders_dir = st.text_input("Location of result folders.", "data/")
    result_folders = list(next(os.walk(result_folders_dir))[1])
    result_folder = st.selectbox("Result folder", result_folders)

    logged_data = load_data(os.path.join("data", result_folder))

    images = logged_data.get_images_with_timestamp('input_frame')
    multi_face_landmark_lists = logged_data.get_data("multi_face_landmarks")

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
        multi_face_landmarks.timestamp
        for multi_face_landmarks in multi_face_landmark_lists
    ]
    # aaa = range(0, )
    predictions = [
        get_state(multi_face_landmark_lists[:i])
        for i in range(1, len(multi_face_landmark_lists))
    ]
    import plotly.graph_objects as go
    fig = go.Figure(
        data=[
            go.Scatter(x=timestamps, y=direction_ys, name="y", line = dict(width=4, dash='dot')),
            go.Scatter(x=timestamps, y=direction_xs, name="x", line = dict(width=4, dash='dot')),
            go.Scatter(x=timestamps, y=predictions, name="preds", line = dict(width=4, dash='dot')),
        ],
    )
    fig.update_layout(xaxis_title='timestamp')
    st.write(fig)



detection_analysis_dashboard()
