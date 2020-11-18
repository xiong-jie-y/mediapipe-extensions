#%%

from pikapi.orientation_estimator import RotationEstimator

rotation_estimator = RotationEstimator(0.70, True)

import numpy as np

rotation_estimator.process_gyro(np.array([-0.00346511, -0.00172339, -0.00172537]), 0.1)
rotation_estimator.process_accel(np.array([0.00913589, -0.02, -9.00]))
rotation_estimator.get_theta()

# %%

# %%
from scipy.spatial.transform.rotation import Rotation
import numpy as np
print(Rotation.from_rotvec([0, 0, -np.pi]).as_quat())
print(Rotation.from_rotvec([0, 0, 0] ).as_quat())
print(Rotation.from_rotvec([0, 0, np.pi]).as_quat())
print(Rotation.from_rotvec([0, 0, 2 * np.pi]).as_quat())
print(Rotation.from_rotvec([0, 0, -2 * np.pi]).as_quat())

# %%
rot1 = Rotation.from_rotvec([0, 0, -np.pi])
rot2 = Rotation.from_rotvec([0, 0, np.pi])
(rot1 * rot2).as_quat()
one_round = (rot2 * rot2)
two_round = (rot2 * rot2 * rot2 * rot2)
four_round = (rot2 * rot2 * rot2 * rot2 * rot2 * rot2 * rot2 * rot2)
# %%
unit_quat = Rotation.from_quat([0,0,0,1])
print(one_round.as_quat())
print(two_round.as_quat())
print(four_round.as_quat())
print((one_round * unit_quat).as_quat())
print((two_round * unit_quat).as_quat())
print((four_round * unit_quat).as_quat())
# %%
