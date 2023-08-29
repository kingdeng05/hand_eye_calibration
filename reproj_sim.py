import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from py_kinetic_backend import Pose3, Rot3, PinholeCameraCal3_S2, Cal3_S2

# from randomised_traj_gen import traj_gen 
from target import CheckerBoardTarget
from rm_factor_graph import calib_rm_factor_graph

np.random.seed(5)
vis_target = False
IMG_WIDTH = 1920 
IMG_HEIGHT = 1200 

def draw_coordinates(T):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Draw the base coordinate frame
    origin = [0, 0, 0]
    ax.quiver(*origin, 1, 0, 0, color='r', length=0.1, label='X_base')
    ax.quiver(*origin, 0, 1, 0, color='g', length=0.1, label='Y_base')
    ax.quiver(*origin, 0, 0, 1, color='b', length=0.1, label='Z_base')

    # Draw the pose coordinate frame transformed by T
    origin_transformed = np.dot(T, np.array([0, 0, 0, 1]))
    x_transformed = np.dot(T, np.array([1, 0, 0, 1]))
    y_transformed = np.dot(T, np.array([0, 1, 0, 1]))
    z_transformed = np.dot(T, np.array([0, 0, 1, 1]))

    ax.quiver(*origin_transformed[:3], *(x_transformed[:3] - origin_transformed[:3]), length=0.1, color='m', label='X_pose')
    ax.quiver(*origin_transformed[:3], *(y_transformed[:3] - origin_transformed[:3]), length=0.1, color='c', label='Y_pose')
    ax.quiver(*origin_transformed[:3], *(z_transformed[:3] - origin_transformed[:3]), length=0.1, color='y', label='Z_pose')

    ax.legend()
    plt.show()

def create_calib_gt():
    # RzRyRx first rotate around z axis, then y axis, lastly x axis.
    calib_rot_gt = Rot3.RzRyRx([-np.pi/2, 0, -np.pi/2])
    calib_t_gt = np.array([0.15, 0, 0])
    calib_gt = Pose3(calib_rot_gt, calib_t_gt).matrix()
    return calib_gt

def create_t2w_gt():
    calib_rot_gt = Rot3.RzRyRx([-np.pi/2, 0, -np.pi/2])
    calib_t_gt = np.array([1.5, 0, 0.5])
    t2w = Pose3(calib_rot_gt, calib_t_gt).matrix()
    return t2w 

def create_intrinsic():
    return np.array([1188, 1188, 0, IMG_WIDTH/2, IMG_HEIGHT/2])

def perturb_by_gaussian(pose, noise, apply_right=True):
    # [rx, ry, rz, x, y, z]
    tf_vec_noise = np.random.normal(np.zeros(6), noise)
    rot = Rot3.RzRyRx(tf_vec_noise[:3])
    tf_noise = Pose3(rot, tf_vec_noise[3:]).matrix()
    if apply_right:
        return pose @ tf_noise 
    else:
        return tf_noise @ pose

def parse_trajectory(traj_q):
    traj_matrix = []
    for pose_q in traj_q:
        pose_q = np.array(pose_q)
        pose = Pose3(Rot3(pose_q[6], pose_q[3], pose_q[4], pose_q[5]), pose_q[:3]).matrix()
        traj_matrix.append(pose)
    return traj_matrix

def main():
    # calib is defined as eye to hand 
    calib_gt = create_calib_gt()
    t2w_gt = create_t2w_gt()
    intrinsic = create_intrinsic()

    calib_noise = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    calib_pert = perturb_by_gaussian(calib_gt, calib_noise)

    t2w_noise = np.array([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])
    t2w_pert = perturb_by_gaussian(t2w_gt, t2w_noise)

    trajs = parse_trajectory(traj_gen())
    pts_target_3d = CheckerBoardTarget(6, 8, 0.1).get_pts_3d()

    pts_all = []
    for h2w in trajs:
        t2e = np.linalg.inv(calib_gt) @ np.linalg.inv(h2w) @ t2w_gt
        k = Cal3_S2(intrinsic)
        
        camera = PinholeCameraCal3_S2(Pose3(np.linalg.inv(t2e)), k) 
        pts_2d = []
        pts_3d = []
        raw_img = np.zeros((IMG_HEIGHT, IMG_WIDTH))
        for pt_3d in pts_target_3d:
            # pt_eye = Pose3(np.linalg.inv(t2e)).transform_to(pt_3d)
            pt_2d = camera.project(pt_3d)
            # pt_2d += np.random.normal([0, 0], [0.1, 0.1])
            pt_2d_int = pt_2d.astype(int)
            cv.circle(raw_img, tuple(pt_2d_int), 2, 255) 
            if pt_2d[0] >= 0 and pt_2d[0] < IMG_WIDTH and pt_2d[1] >= 0 and pt_2d[1] < IMG_HEIGHT: 
                pts_2d.append(pt_2d)
                pts_3d.append(pt_3d)
        if vis_target:
            cv.imshow("test", raw_img)
            cv.waitKey(0)
        pts_all.append({
            "2d": pts_2d,
            "3d": pts_3d
        })

    intrinsic_pert = intrinsic + np.array([10, 20, 0, 8, 7])
    calib_ret, t2w_ret = calib_rm_factor_graph(calib_pert, t2w_pert, intrinsic_pert, trajs, pts_all)
    print(Pose3.logmap(Pose3(calib_gt).between(Pose3(calib_ret))))
    print(Pose3.logmap(Pose3(calib_gt).between(Pose3(calib_pert))))


if __name__ == "__main__":
    main()
    # draw_coordinates(create_calib_gt())