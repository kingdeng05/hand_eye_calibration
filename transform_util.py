import numpy as np

from py_kinetic_backend import Pose3, Rot3

def euler_vec_to_mat(vec):
    return Pose3(Rot3.rzryrx(vec[:3]), vec[3:]).matrix()

def mat_to_euler_vec(mat, use_deg=True):
    pose = Pose3(mat)
    rot_vec = pose.rotation().xyz()
    if use_deg:
        rot_vec = np.rad2deg(rot_vec)
    rot_vec = rot_vec.tolist()
    trans_vec = pose.translation().tolist()
    return np.array(rot_vec + trans_vec) 

