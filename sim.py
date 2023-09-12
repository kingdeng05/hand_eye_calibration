import numpy as np

from py_kinetic_backend import Pose3, Rot3, PinholeCameraCal3DS2, Cal3DS2

IMG_WIDTH = 2048 
IMG_HEIGHT = 1536 
FOCAL_LENGTH = {
    "6mm": 1739,
    "4mm": 1159
}

def create_calib_gt():
    # rzryrx first rotate around z axis, then y axis, lastly x axis.
    calib_rot_gt = Rot3.rzryrx([-np.pi/2, 0, -np.pi/2])
    calib_t_gt = np.array([0.13, 0, 0])
    calib_gt = Pose3(calib_rot_gt, calib_t_gt).matrix()
    return calib_gt

def create_t2w_gt():
    calib_rot_gt = Rot3.rzryrx([-np.pi/2, 0, -np.pi/2])
    calib_t_gt = np.array([3, 0, 0.8])
    t2w = Pose3(calib_rot_gt, calib_t_gt).matrix()
    return t2w 

def create_cube_t2w_gt():
    calib_rot_gt = Rot3.rzryrx([-np.pi/2, 0, -np.pi/2])
    calib_t_gt = np.array([3.8, 0, 0.143])
    t2w = Pose3(calib_rot_gt, calib_t_gt).matrix()
    return t2w 

def create_intrinsic():
    return np.array([FOCAL_LENGTH, FOCAL_LENGTH, 0, IMG_WIDTH/2, IMG_HEIGHT/2])

def create_intrinsic_distortion(focal_length="6mm"):
    focal_length_xy = FOCAL_LENGTH[focal_length]
    return np.array([focal_length_xy, focal_length_xy, 0, IMG_WIDTH/2, IMG_HEIGHT/2, 0, 0, 0, 0])

