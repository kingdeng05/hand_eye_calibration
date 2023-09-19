import os
import yaml
import random
import numpy as np
import cv2 as cv 
from collections import defaultdict
from py_kinetic_backend import Pose3, Rot3, PinholeCameraCal3DS2, Cal3DS2
from matplotlib import pyplot as plt

from read_rosbag import read_handeye_bag
from rm_factor_graph import calib_rm_factor_graph, calib_rm2_factor_graph
from dhe_factor_graph import calibrate_dhe_factor_graph 
from reproj_sim import create_calib_gt, create_cube_t2w_gt
from target import ArucoCubeTarget, ArucoBoardTarget
from aruco_detector import ArucoDetector
from calibrate_intrinsic import calibrate_intrinsic, calculate_reproj_error, reprojection_plot

random.seed(5)
np.set_printoptions(precision=3, suppress=True)


def pose_msg_to_tf(pose_msg):
    rot = Rot3(pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z)
    return Pose3(rot, np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])).matrix()

def Cal3DS2_to_KD(intr):
    return np.array([[intr[0], intr[2], intr[3]], [0, intr[1], intr[4]], [0, 0, 1]]), intr[5:].reshape(-1, 4)

def tf_to_vec(tf, use_degree=True):
    pose = Pose3(tf)
    rot_vec = pose.rotation().xyz()
    if use_degree:
        rot_vec = np.rad2deg(rot_vec)
    rot_vec = rot_vec.tolist()
    trans_vec = pose.translation().tolist()
    return np.array(rot_vec + trans_vec) 

def calibrate_hand_eye_rm(bag_path):
    calib_init = create_calib_gt()
    t2w_init = create_cube_t2w_gt()
    # intrinsic = np.array([1742.399, 1741.498, 0, 1030.966, 782.361, -0.28, 0.15, -0.002, -0.001])
    # intrinsic = np.array([1737.045, 1736.33, 0, 1021.933, 781.133, -0.282, 0.146, -0.002, 0.])
    # intrinsic = np.array([1742.295, 1741.466, 0, 1031.074, 782.305, -0.28, 0.15, -0.001, -0.001])
    intrinsic = calibrate_intrinsic(bag_path)

    # target = ArucoBoardTarget(5, 5, 0.166, 0.033, 100)
    target = ArucoCubeTarget(1.035)
    detector = ArucoDetector(vis=False)
    pts_all = []
    poses_all = []
    for img, pose_msg, _, _, _ in read_handeye_bag(bag_path):
        pts = defaultdict(list) 
        corners, ids = detector.detect(img)
        if len(ids) == 0:
            continue
        for corner, id in zip(corners, ids):
            pts_3d = target.find_3d_pts_by_id(id)
            if pts_3d is None:
                continue
            for pt_2d, pt_3d in zip(corner, pts_3d):
                pts["2d"].append(pt_2d) 
                pts["3d"].append(pt_3d)
        poses_all.append(pose_msg_to_tf(pose_msg))
        pts_all.append(pts)

    # solve for parameters
    calib_ret, t2w_ret, intrinsic_ret, poses_ret = calib_rm_factor_graph(calib_init, t2w_init, intrinsic, poses_all, pts_all)

    # print changes compared to initial values
    print("".join(["*" for _ in range(30)]))
    print("calib ret:\n", tf_to_vec(calib_ret))
    print("t2w ret:\n", tf_to_vec(t2w_ret))
    print("intrinsic ret:\n", intrinsic_ret)
    print("calib ret diff:\n", tf_to_vec(Pose3(calib_init).between(Pose3(calib_ret)).matrix()))
    print("t2w ret diff:\n", tf_to_vec(Pose3(t2w_init).between(Pose3(t2w_ret)).matrix()))
    print("intrinsic diff:\n", intrinsic_ret - intrinsic)
    print("".join(["*" for _ in range(30)]))

    # print out the hand poses differences 
    # poses_diff = []
    # for pose_opt, pose_init in zip(poses_ret, poses_all):
    #     poses_diff.append(tf_to_vec(Pose3(pose_init).between(Pose3(pose_opt)).matrix()))
    # poses_diff = np.array(poses_diff)
    # _, axes = plt.subplots(6, 1, squeeze=True)
    # for i in range(6): 
    #     axes[i].plot(poses_diff[:, i])
    #     axes[i].set_title(f"avg: {poses_diff[:, i].mean()}, std: {poses_diff[:, i].std()}")
    # plt.show()

    # visual evaluation 
    pts_all_2d = []
    pts_all_proj = []
    for idx, (img, pose_msg, _, _, _) in enumerate(read_handeye_bag(bag_path)):
        pts = defaultdict(list) 
        corners, ids = detector.detect(img)
        if len(ids) == 0:
            continue
        for corner, id in zip(corners, ids):
            pts_3d = target.find_3d_pts_by_id(id)
            if pts_3d is None:
                continue
            for pt_2d, pt_3d in zip(corner, pts_3d):
                pts["2d"].append(pt_2d) 
                pts["3d"].append(pt_3d)
        poses_all.append(pose_msg_to_tf(pose_msg))
        pts_all.append(pts)
        # projection 
        c2t = np.linalg.inv(t2w_ret) @ poses_ret[idx] @ calib_ret
        camera = PinholeCameraCal3DS2(Pose3(c2t), Cal3DS2(intrinsic_ret))
        img_copy = img.copy()
        pts_proj = []
        for pt_2d, pt_3d in zip(pts["2d"], pts["3d"]): 
            pt_proj = camera.project(pt_3d).astype(int)
            cv.circle(img_copy, tuple(pt_2d.astype(int)), 4, (0, 255, 0))
            cv.circle(img_copy, tuple(pt_proj), 4, (0, 0, 255))
            cv.line(img_copy, tuple(pt_proj), tuple(pt_2d.astype(int)), (255, 0, 0), 3)
            pts_proj.append(pt_proj)
            pts_all_2d.append(pt_2d)
            pts_all_proj.append(pt_proj)
        # visualization
        pts_proj = np.array(pts_proj)
        if len(pts_proj):
            error_cur = calculate_reproj_error(pts_proj, pts["2d"])
            cv.putText(
                img_copy, 
                f"reproj_error: {error_cur}",
                (200, 200),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0)
            )
        img_copy = cv.resize(img_copy, (int(img_copy.shape[1]/2), int(img_copy.shape[0]/2)))
        cv.imshow("proj", img_copy)
        cv.waitKey(0)

    # calculate overall reprojection error
    pts_all_2d = np.array(pts_all_2d)
    pts_all_proj = np.array(pts_all_proj)
    print("reprojection error: ", calculate_reproj_error(pts_all_2d, pts_all_proj))
    reprojection_plot(pts_all_2d, pts_all_proj)

    return calib_ret, poses_ret 

def calibrate_hand_eye_dhe(bag_path):
    calib_init = create_calib_gt()
    intrinsic = calibrate_intrinsic(bag_path)
    target = ArucoBoardTarget(5, 5, 0.166, 0.033, 100)
    detector = ArucoDetector(vis=False)
    hand_poses = []
    eye_poses = []
    for img, pose_msg, _, _, _ in read_handeye_bag(bag_path):
        corners, ids = detector.detect(img)
        if len(ids) == 0:
            continue
        pts_3d = []
        pts_2d = []
        for corner, id in zip(corners, ids):
            pts_target = target.find_3d_pts_by_id(id)
            if pts_target is None:
                continue
            for pt_2d, pt_3d in zip(corner, pts_target):
                pts_3d.append(pt_3d) 
                pts_2d.append(pt_2d)
        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)
        K, D = Cal3DS2_to_KD(intrinsic)
        _, rvec, tvec = cv.solvePnP(pts_3d, pts_2d, K, D)
        R = cv.Rodrigues(rvec)[0]
        eye_pose = Pose3(Rot3(R), tvec).matrix()
        eye_poses.append(eye_pose)
        hand_poses.append(pose_msg_to_tf(pose_msg))

    zipped = list(zip(hand_poses, eye_poses))
    random.shuffle(zipped)
    hand_poses, eye_poses = zip(*zipped)

    R_hand2base = [hand_pose[:3, :3] for hand_pose in hand_poses]
    t_hand2base = [hand_pose[:3, 3] for hand_pose in hand_poses]
    R_target2cam = [eye_pose[:3, :3] for eye_pose in eye_poses]
    t_target2cam = [eye_pose[:3, 3] for eye_pose in eye_poses]
    R_eye2hand, t_eye2hand = cv.calibrateHandEye(R_hand2base, t_hand2base, R_target2cam, t_target2cam, cv.CALIB_HAND_EYE_PARK)
    calib_park = Pose3(Rot3(R_eye2hand), t_eye2hand)
    print("closed form solution: ", tf_to_vec(calib_park.matrix()))
    print("calib diff with closed form: ", tf_to_vec(Pose3(calib_init).between(calib_park).matrix()))

    calib_opt = calibrate_dhe_factor_graph(calib_init, hand_poses, eye_poses)
    print("least squares solution: ", tf_to_vec(calib_opt))
    print("calib diff: ", tf_to_vec(Pose3(calib_init).between(Pose3(calib_opt)).matrix()))

def calibrate_hand_eye_rm2(bag_path):
    calib_init = create_calib_gt()
    t2w_init = create_cube_t2w_gt()
    intrinsic = calibrate_intrinsic(bag_path)

    target = ArucoCubeTarget(1.035, use_ids=(50,))
    detector = ArucoDetector(vis=False)
    pts_all = []
    hand_poses = []
    eye_poses = []
    for img, pose_msg, _, _, _ in read_handeye_bag(bag_path):
        pts = defaultdict(list) 
        corners, ids = detector.detect(img)
        if len(ids) == 0:
            continue
        pts_3d, pts_2d = [], []
        for corner, id in zip(corners, ids):
            pts_target = target.find_3d_pts_by_id(id)
            if pts_target is None:
                continue
            for pt_2d, pt_3d in zip(corner, pts_target):
                pts["2d"].append(pt_2d) 
                pts["3d"].append(pt_3d)
                pts_3d.append(pt_3d) 
                pts_2d.append(pt_2d)
        
        # collect all the observations
        pts_all.append(pts)

        # compute initial value for eye poses
        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)
        K, D = Cal3DS2_to_KD(intrinsic)
        _, rvec, tvec = cv.solvePnP(pts_3d, pts_2d, K, D)
        R = cv.Rodrigues(rvec)[0]
        eye_pose = Pose3(Rot3(R), tvec).matrix()
        eye_poses.append(eye_pose)

        # append hand poses
        hand_poses.append(pose_msg_to_tf(pose_msg))

    # solve for parameters
    calib_ret, t2w_ret, intrinsic_ret, cam_poses_ret = calib_rm2_factor_graph(calib_init, t2w_init, intrinsic, eye_poses, hand_poses, pts_all)

    # print changes compared to initial values
    print("".join(["*" for _ in range(30)]))
    print("calib ret:\n", tf_to_vec(calib_ret))
    print("t2w ret:\n", tf_to_vec(t2w_ret))
    print("intrinsic ret:\n", intrinsic_ret)
    print("calib ret diff:\n", tf_to_vec(Pose3(calib_init).between(Pose3(calib_ret)).matrix()))
    print("t2w ret diff:\n", tf_to_vec(Pose3(t2w_init).between(Pose3(t2w_ret)).matrix()))
    print("intrinsic diff:\n", intrinsic_ret - intrinsic)
    print("".join(["*" for _ in range(30)]))

    # visual evaluation 
    pts_all_2d = []
    pts_all_proj = []
    for idx, (img, pose_msg, _, _, _) in enumerate(read_handeye_bag(bag_path)):
        pts = defaultdict(list) 
        corners, ids = detector.detect(img)
        if len(ids) == 0:
            continue
        for corner, id in zip(corners, ids):
            pts_3d = target.find_3d_pts_by_id(id)
            if pts_3d is None:
                continue
            for pt_2d, pt_3d in zip(corner, pts_3d):
                pts["2d"].append(pt_2d) 
                pts["3d"].append(pt_3d)
        hand_poses.append(pose_msg_to_tf(pose_msg))
        pts_all.append(pts)
        # projection 
        c2t = np.linalg.inv(cam_poses_ret[idx]) 
        camera = PinholeCameraCal3DS2(Pose3(c2t), Cal3DS2(intrinsic_ret))
        img_copy = img.copy()
        pts_proj = []
        for pt_2d, pt_3d in zip(pts["2d"], pts["3d"]): 
            pt_proj = camera.project(pt_3d).astype(int)
            cv.circle(img_copy, tuple(pt_2d.astype(int)), 4, (0, 255, 0))
            cv.circle(img_copy, tuple(pt_proj), 4, (0, 0, 255))
            cv.line(img_copy, tuple(pt_proj), tuple(pt_2d.astype(int)), (255, 0, 0), 3)
            pts_proj.append(pt_proj)
            pts_all_2d.append(pt_2d)
            pts_all_proj.append(pt_proj)
        # visualization
        # pts_proj = np.array(pts_proj)
        # if len(pts_proj):
        #     error_cur = calculate_reproj_error(pts_proj, pts["2d"])
        #     cv.putText(
        #         img_copy, 
        #         f"reproj_error: {error_cur}",
        #         (200, 200),
        #         cv.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (0, 255, 0)
        #     )
        # img_copy = cv.resize(img_copy, (int(img_copy.shape[1]/2), int(img_copy.shape[0]/2)))
        # cv.imshow("proj", img_copy)
        # cv.waitKey(0)

    # calculate overall reprojection error
    pts_all_2d = np.array(pts_all_2d)
    pts_all_proj = np.array(pts_all_proj)
    print("reprojection error: ", calculate_reproj_error(pts_all_2d, pts_all_proj))
    reprojection_plot(pts_all_2d, pts_all_proj)


if __name__ == "__main__":
    # bag_name = "/home/fuhengdeng/hand_eye.bag"
    bag_name = "/home/fuhengdeng/fuheng.bag"
    # calibrate_hand_eye_dhe(bag_name)
    # calibrate_hand_eye_rm(bag_name)
    calibrate_hand_eye_rm2(bag_name)