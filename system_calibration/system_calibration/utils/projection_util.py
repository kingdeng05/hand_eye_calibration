import numpy as np
import cv2 as cv

from py_kinetic_backend import Pose3, Rot3, PinholeCameraCal3DS2, Cal3DS2
from py_kinetic_backend import PinholeCameraCal3Rational, Cal3Rational

def create_camera(tf_cam_pose, intrinsic_vec, model="Cal3Rational"):
    if model == "Cal3DS2":
        cam = PinholeCameraCal3DS2(Pose3(tf_cam_pose), Cal3DS2(intrinsic_vec))
    elif model == "Cal3Rational":
        cam = PinholeCameraCal3Rational(Pose3(tf_cam_pose), Cal3Rational(intrinsic_vec))
    else:
        raise NotImplementedError(f"model {model} not supported")
    return cam

def transfer_3d_pts_to_img(pts_3d, tf_cam_pose, intrinsic_vec, model="Cal3Rational"):
    cam = create_camera(tf_cam_pose, intrinsic_vec, model)
    pts_proj = []
    for pt_3d in pts_3d:
        pts_proj.append(cam.project(pt_3d))
    return np.array(pts_proj)

def calculate_reproj_error(pts_2d, pts_proj):
    return np.linalg.norm(pts_2d - pts_proj, axis=1).mean()

def draw_pts_on_img(img, pts_2d, s=2, c=(0, 255, 0)):
    img_vis = img.copy()
    for pt in pts_2d.astype(int):
        cv.circle(img_vis, tuple(pt), s, c)
    return img_vis
         
    