from collections import deque
import facelandmark_utils as flu

WINDOW_PERIOD_S_ = 0.5
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

def get_state(face_landmark_lists, center=None):
    xmax, xmin, ymax, ymin = get_minmax_feature(face_landmark_lists, center)

    if xmax is None:
        return 0

    if ymax > 0.2 and ymin <= 0.0:
        return 1
    elif xmax > 0.2 and xmin < -0.2:
        return -1
    else:
        return 0

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

        return get_state(self.face_landmark_lists, center=self.initial_direction)