import os
import numpy as np
import cv2 as cv
from copy import deepcopy 

from system_calibration.simulation import *
from system_calibration.frontend import ArucoDetector
from system_calibration.backend import solve_base_to_tt_graph_2 
from system_calibration.utils import euler_vec_to_mat, mat_to_euler_vec
from system_calibration.utils import transfer_3d_pts_to_img, calculate_reproj_error

from build_sim_sys import build_sim_sys, simulate_projection, read_cam_intrinsic, read_cam2ee_calib

VIS = False 

np.random.seed(5)
np.set_printoptions(precision=3, suppress=True)

def print_reproj_stats(pts_2d, pts_proj):
    local_re = []
    for pts_2d_frame, pts_proj_frame in zip(pts_2d, pts_proj):
        err_frame = calculate_reproj_error(pts_2d_frame, pts_proj_frame)
        local_re.append(err_frame)
    global_re = np.linalg.norm(np.vstack(pts_2d) - np.vstack(pts_proj), axis=1)
    print(f"local reproj error:\n  mean: {np.mean(local_re)}, min: {np.min(local_re)}, max: {np.max(local_re)}, std: {np.std(local_re)}")
    print(f"global reproj error:\n  mean: {np.mean(global_re)}, min: {np.min(global_re)}, max: {np.max(global_re)}, std: {np.std(global_re)}")

def print_vec(mat):
    print(mat_to_euler_vec(mat, use_deg=True))

def get_gt():
    sim = build_sim_sys()
    gt = dict()
    gt["track2tt"] = sim.calibration["track"]["tt"] 
    gt["base2track"] = sim.calibration["robot"]["track"] 
    gt["intrinsic"] = read_cam_intrinsic()
    gt["cam2ee"] = read_cam2ee_calib() 
    gt["target2tt_0"] = sim.calibration["cube"]["tt"] 
    return gt

def sim_bag_read():
    sim = build_sim_sys()
    ee_pose_vec = [0, 0, 0, 0.6, 0, 0.8]
    sim.move("robot", ee_pose_vec) 
    for tr in [0.5, 1, 2]:
        sim.move("track", tr)
        for tt in np.linspace(0, 2 * np.pi, 7):
            sim.move("tt", tt)
            pts_2d, pts_3d = sim.capture("camera")
            if VIS:
                simulate_projection(pts_2d)
            yield pts_3d, pts_2d, euler_vec_to_mat(ee_pose_vec), track_reading_to_transform(tr), tt_reading_to_transform(tt) 

def read_bag():
    pass

def perturb_pose3(pose_mat, var, current=True):
    assert(len(var) == 6)
    pert_vec = [np.random.normal(0, v) for v in var]
    pert_mat = euler_vec_to_mat(pert_vec)
    if current:
        return pose_mat @ pert_mat  
    else:
        return pert_mat @ pose_mat

def perturb_pts(pts, var):
    # pts must be in (n, 2) or (n, 3)
    assert pts.shape[1] == len(var)
    noises = []
    for v in var:
        noises.append(np.random.normal(0, v, len(pts)))
    noises = np.array(noises).T
    return pts + noises 

def track_reading_to_transform(track_reading):
    # track moves in x direction, 0 is at the very begining
    # so only negative values
    tf = np.eye(4)
    # take negative sign here because x is pointing forward but track moves backward
    tf[0, 3] = -track_reading 
    return tf

def tt_reading_to_transform(tt_reading):
    # tt moves in yaw direction, 0 is at the very begining
    # so only positive values without wrap around
    # assuming no negative value will be here
    assert(tt_reading >= 0)
    while tt_reading >= np.pi * 2:
        tt_reading -= np.pi * 2 
    vec = [0, 0, tt_reading, 0, 0, 0]
    return euler_vec_to_mat(vec) 

def append_dict_list(d, key, val):
    if key not in d:
        d[key] = [] 
    d[key].append(val)

def get_data(bag_path):
    # read data in either sim mode or actual bag
    if bag_path is not None and os.path.exists(bag_path):
        print(f"Loading bag from {bag_path}")
        read = read_bag(bag_path)
    else:
        print(f"No bag path is given or {bag_path} doesn't exist, running simulation...")
        read = sim_bag_read()
    return read
    
def calibrate_base_to_tt(bag_path=None, perturb=False):
    hand_poses = []
    track_tfs = []
    tt_tfs = []
    pts_all = []
    initials = get_gt()

    for _, (pts_3d, pts_2d, hand_pose, track_tf, tt_tf) in enumerate(get_data(bag_path)):
        track_tfs.append(track_tf)
        tt_tfs.append(tt_tf)
        hand_poses.append(hand_pose) 
        pts_pair = dict()
        pts_pair["3d"] = pts_3d
        pts_pair["2d"] = pts_2d
        pts_all.append(pts_pair)

    # perturbation test
    if perturb:
        pert = deepcopy(initials)
        pert["track2tt"] = perturb_pose3(pert["track2tt"], [1e-3, 1e-3, 0, 0.1, 0.1, 1e-2]) 
        pert["base2track"] = perturb_pose3(pert["base2track"], [1e-5, 1e-5, 0.1, 0, 0, 0]) 
        track_tfs = [perturb_pose3(tr_tf, [0, 0, 0, 1e-2, 0, 0]) for tr_tf in track_tfs]
        tt_tfs = [perturb_pose3(tt_tf, [0, 0, 2e-3, 0, 0, 0]) for tt_tf in tt_tfs]
        hand_poses = [perturb_pose3(hand_pose, [1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]) for hand_pose in hand_poses]
        pts_all = [{"2d": perturb_pts(pts["2d"], [1, 1]), "3d": pts["3d"]} for pts in pts_all] 
        pert["target2tt"] = perturb_pose3(pert["track2tt"], [1e-2, 1e-5, 1e-5, 1e-4, 1e-1, 1e-1])
    else:
        pert = initials

    print("frontend features collection finished, running backend...")
    tf_track2tt, tf_base2track, tf_ee2base, tf_target2tt, track_tfs, tt_tfs = solve_base_to_tt_graph_2(pts_all, hand_poses, track_tfs, tt_tfs, pert)
    print("diff of track2tt: ", mat_to_euler_vec(np.linalg.inv(initials["track2tt"]) @ tf_track2tt))
    print("diff of base to track: ", mat_to_euler_vec(np.linalg.inv(initials["base2track"]) @ tf_base2track))
    print("diff of ee to base: ", mat_to_euler_vec(np.linalg.inv(hand_poses[0]) @ tf_ee2base))
    print("diff of target to tt: ", mat_to_euler_vec(np.linalg.inv(initials["target2tt_0"]) @ tf_target2tt))

    # calculate reprojection error
    pts_2d_all = [] 
    pts_proj_all = []
    for _, ((pts_3d, pts_2d, _, _, _), track_tf, tt_tf) in enumerate(zip(get_data(bag_path), track_tfs, tt_tfs)):
        tf_target2tt_i = tt_tf @ tf_target2tt
        tf_cam2target = np.linalg.inv(tf_target2tt_i) @ tf_track2tt @ track_tf @ tf_base2track @ tf_ee2base @ initials["cam2ee"] 
        pts_proj = transfer_3d_pts_to_img(pts_3d, tf_cam2target, initials["intrinsic"])
        pts_2d_all.append(pts_2d)
        pts_proj_all.append(pts_proj)
    print_reproj_stats(pts_2d_all, pts_proj_all)


if __name__ == "__main__":
    calibrate_base_to_tt(bag_path=None, perturb=True)

