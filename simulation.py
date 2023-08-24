import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from dhe_factor_graph import calibrate_dhe_factor_graph 
# from py_kinetic_factors import calibrate_dhe_factor_graph 

np.random.seed(5)
np.set_printoptions(precision=4, suppress=True)


def generate_se3_perturbation(sigma):
    # Generate a 6D vector of Gaussian noise
    noise = np.random.normal(0, sigma, 6)
    
    # Extract rotational and translational parts
    omega = noise[:3]
    v = noise[3:]
    
    # Convert the rotation vector to a rotation matrix using the exponential map
    rotation = R.from_rotvec(omega).as_matrix()
    
    # Create SE(3) perturbation matrix
    perturbation = np.eye(4)
    perturbation[:3, :3] = rotation
    perturbation[:3, 3] = v
    
    return perturbation


def apply_perturbations(traj, noise):
    traj_pert = []
    for pose in traj:
        traj_pert.append(generate_se3_perturbation(noise) @ pose)
    return traj_pert


# Function to generate a random rotation matrix
def random_rotation_matrix():
    theta = np.random.uniform(0, 2*np.pi)  # Rotation angle
    axis = np.random.uniform(-1, 1, 3)  # Rotation axis
    axis /= np.linalg.norm(axis)  # Normalize to unit vector

    # Create the rotation matrix using axis-angle representation
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


# Function to generate a random translation vector
def random_translation_vector(scale=1.0):
    return np.random.uniform(-scale, scale, 3)


# Function to generate a 4x4 transformation matrix given R and t
def generate_transformations(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

n_poses = 50 
R_calib = random_rotation_matrix()
t_calib = random_translation_vector()
T_calib = generate_transformations(R_calib, t_calib)

trajectory_A, trajectory_B = [], []
for i in range(n_poses):
    R_A = random_rotation_matrix()
    t_A = random_translation_vector()
    T_A = generate_transformations(R_A, t_A)

    # Compute T_B = T_A * T_calib
    T_B = T_A.dot(T_calib)

    trajectory_A.append(T_A)
    trajectory_B.append(np.linalg.inv(T_B))

# apply gaussian perturbation
trajectory_A = apply_perturbations(trajectory_A, np.append(np.deg2rad([0.1, 0.1, 0.1]), [0.1, 0.1, 0.1]))
trajectory_B = apply_perturbations(trajectory_B, np.append(np.deg2rad([0.1, 0.1, 0.1]), [0.1, 0.1, 0.1]))

# Convert to OpenCV's format (Rodrigues vector for rotation + translation vector)
rot_A_rodri = [tf[:3, :3] for tf in trajectory_A]
rot_B_rodri = [tf[:3, :3] for tf in trajectory_B]
t_A = [tf[:3, 3] for tf in trajectory_A]
t_B = [tf[:3, 3] for tf in trajectory_B]

# # Prepare containers for results
rvecs, tvecs = {}, {} 
methods = {
    "Tsai": cv2.CALIB_HAND_EYE_TSAI,
    "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    "Park": cv2.CALIB_HAND_EYE_PARK,
}

# Perform hand-eye calibration using different methods
for method_name, method in methods.items():
    rvec, tvec = cv2.calibrateHandEye(
        rot_A_rodri, t_A,
        rot_B_rodri, t_B,
        method=method)
    rvecs[method_name] = rvec
    tvecs[method_name] = tvec.flatten()

tvec_gt = T_calib[:3, 3]

for method, rot in rvecs.items():
    print(f"{method}")
    t = tvecs[method]
    print("rot diff: ", np.rad2deg(cv2.Rodrigues(rot.T.dot(T_calib[:3, :3]))[0].flatten()))
    print("t diff: ", t - T_calib[:3, 3])

# test out the factor
print("using DHE factor graph")
calibrate_dhe_factor_graph(T_calib, trajectory_A, trajectory_B)
    
