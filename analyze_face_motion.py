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

    # Choose result dir.
    result_folders_dir = st.text_input("Location of result folders.", "data/")
    result_folders = list(next(os.walk(result_folders_dir))[1])
    result_folder = st.selectbox("Result folder", result_folders)

    logged_data = load_data(os.path.join("data", result_folder))

    images = logged_data.get_images_with_timestamp('input_frame')
    multi_face_landmark_lists = logged_data.get_data("multi_face_landmarks")

    assert(len(images) == len(multi_face_landmark_lists))

    frame_index = st.slider("Index", 0, len(images))

    # st.write(type(images[0].data))
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
    import plotly.graph_objects as go
    fig = go.Figure(
        data=[
            go.Scatter(x=timestamps, y=direction_ys),
            go.Scatter(x=timestamps, y=direction_xs),
        ],
        layout_title_text="A Figure Displayed with fig.show()"
    )
    st.write(fig)



detection_analysis_dashboard()
