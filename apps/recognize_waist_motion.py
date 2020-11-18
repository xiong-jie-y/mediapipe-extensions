from pikapi.motion.complimentary_filter import ComplimentaryFilterOrientationEstimator
from pikapi.gui.pose_visualization import PoseVisualizer
from pikapi.formats.basics import IMUInfo
import numpy as np
from pikapi.devices.realsense import RGBDWithIMU
from pikapi.orientation_estimator import RotationEstimator
from scipy.spatial.transform.rotation import Rotation

# rotation_estimator = RotationEstimator(0.7, True)
# rotation_estimator = ComplimentaryFilterOrientationEstimator(1.0)
rotation_estimator = ComplimentaryFilterOrientationEstimator(0.7)

device = RGBDWithIMU()

from collections import deque
acc_vectors = deque([])

visualizer = PoseVisualizer()

def rpy_to_rotation(pyr):
    rot_axises = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    rots = [Rotation.from_rotvec(ax * ad) for ax, ad in zip(rot_axises, pyr)]
    all_rot = rots[2] * rots[1] * rots[0]
    
    return all_rot

while True:
    color_frame, depth_frame, acc, gyro, timestamp = device.get_data()
    print("Raw Data", acc)
    
    # 
    acc_vector = np.array([-acc.x, -acc.y, acc.z])
    rotation_estimator.process_gyro(
        np.array([gyro.x, gyro.y, gyro.z]), timestamp)
    print("Gyro", gyro)
    print("Acc", acc)
    rotation_estimator.process_accel(acc_vector)
    # theta = rotation_estimator.get_theta()
    pose = rotation_estimator.get_pose()
    print("Rotation Estimator", pose)

    # 
    acc_size = np.linalg.norm(acc_vector)
    print(f"Norm {acc_size}.")
    acc_vectors.append(acc_vector)
    if len(acc_vectors) > 200:
        acc_vectors.popleft()
    mean_acc = np.mean(acc_vectors, axis=0)
    imu_info = IMUInfo(acc=acc)
    print("Mean ACC", mean_acc)

    acc_vector = -acc_vector
    # direction = rpy_to_rotation(theta).apply([1,0, 0])
    direction = Rotation.from_quat(pose).apply([0,0, 1])
    direction = np.array([direction[0], direction[1], direction[2]])
    # acc_vector[0] *= -1
    # acc_vector[1] *= -1
    visualizer.update(0, acc_vector)
    visualizer.update(1, direction)
    # visualizer.update(mean_acc)