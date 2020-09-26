#%%
import os
root_path = os.path.expanduser("~/gitrepos/pikapi")
import sys; sys.path.append(root_path)

import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=10000)

import pickle
from pikapi.utils.landmark import get_3d_center, get_camera_coord_landmarks
args = pickle.load(open(os.path.join(root_path, "get_3d_center_args.pkl"), 'rb'))
print(args)
face_landmark, width, height, depth_image, intrinsic_matrix = args
get_3d_center(*args)
# %%
a = get_camera_coord_landmarks(face_landmark, width, height, depth_image, intrinsic_matrix)
#%%
a
# %%
face_landmark[:, 0] * width - intrinsic_matrix.ppx
# %%
800 / intrinsic_matrix.fx
# %%
from pikapi.utils.realsense import get_intrinsic
get_intrinsic(640, 360)
# %%
