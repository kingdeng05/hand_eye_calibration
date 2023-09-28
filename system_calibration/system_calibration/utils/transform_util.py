import numpy as np

from py_kinetic_backend import Pose3, Rot3

def euler_vec_to_mat(vec, use_deg=False):
    if use_deg:
        vec[:3] = np.deg2rad(vec[:3])
    return Pose3(Rot3.rzryrx(vec[:3]), vec[3:]).matrix()

def mat_to_euler_vec(mat, use_deg=True):
    pose = Pose3(mat)
    rot_vec = pose.rotation().xyz()
    if use_deg:
        rot_vec = np.rad2deg(rot_vec)
    rot_vec = rot_vec.tolist()
    trans_vec = pose.translation().tolist()
    return np.array(rot_vec + trans_vec) 

def mat_to_quaternion(pose):
    quat = Rot3(pose[:3, :3]).quaternion()  # qw, qx, qy, qz
    return pose[:3, 3].tolist() + quat[[1, 2, 3, 0]].tolist()

def transform_3d_pts(pts, tf):
    pts_tf = []
    for pt in pts: 
        pt_tf = Pose3(tf).transform_from(pt)
        pts_tf.append(pt_tf)
    return np.array(pts_tf).reshape(-1, 3)



    


