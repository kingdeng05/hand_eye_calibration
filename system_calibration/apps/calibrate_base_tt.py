import os
import numpy as np
import cv2 as cv
from copy import deepcopy 

from system_calibration.simulation import *
from system_calibration.simulation.target import ArucoCubeTarget
from system_calibration.frontend import ArucoDetector
from system_calibration.backend import solve_base_to_tt_graph_2 
from system_calibration.utils import euler_vec_to_mat, mat_to_euler_vec
from system_calibration.utils import transfer_3d_pts_to_img, calculate_reproj_error

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

def sim_bag_read(gt):
    # make sure using the same seed to make the results the same
    np.random.seed(10)
    track2tt = gt["track2tt"]
    base2track = gt["base2track"]
    cam2ee = gt["cam2ee"]
    intrinsic = gt["intrinsic"]
    target2tt = gt["target2tt_0"]
    target = ArucoCubeTarget(1.035)
   
    # simulate the movement of different components
    tt_meas_cnt = 5
    track_meas_cnt = 1 
    tt_readings = np.linspace(0, 2 * np.pi, tt_meas_cnt)
    track_readings = np.random.uniform(0, 1.5, track_meas_cnt)

    # fix ee to base for now
    ee2base = euler_vec_to_mat([0, 0, 0, 0.6, 0, 0.8])
    for tr in track_readings:
        track_tf = track_reading_to_transform(tr)
        cam2tt = track2tt @ track_tf @ base2track @ ee2base @ cam2ee 
        for tt in tt_readings:
            tt_tf = tt_reading_to_transform(tt)
            target2tt_i = tt_tf @ target2tt
            target2cam = np.linalg.inv(cam2tt) @ target2tt_i 
            cam = VisibleCamera(intrinsic) 
            pts_2d, pts_3d = cam.project(np.linalg.inv(target2cam), target, (IMG_WIDTH, IMG_HEIGHT))
            if VIS:
                img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
                for pt in pts_2d.astype(int):
                    cv.circle(img, tuple(pt), 2, (0, 255, 0))
                cv.putText(
                    img, 
                    f"track meas: {tr}m, tt angle: {np.rad2deg(tt)}deg",
                    (100, 100),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0)
                )
                img = cv.resize(img, (int(IMG_WIDTH/2), int(IMG_HEIGHT/2)))
                cv.imshow("view", img) 
                cv.waitKey(0)
            yield pts_3d, pts_2d, ee2base, track_tf, tt_tf

def read_bag():
    pass

def sim_ground_truth():
    gt = dict()
    gt["track2tt"] = euler_vec_to_mat([0, 0, -180, 3.71, 0, 0.38], use_deg=True)
    gt["base2track"] = euler_vec_to_mat([0, 0, 1, 0, 0, 0], use_deg=True)
    gt["cam2ee"] = euler_vec_to_mat([-90.456, -0.105, -89.559, 0.131, 0.002, 0], use_deg=True)
    gt["intrinsic"] = [1180.976, 1178.135, 0., 1033.019, 796.483, -0.211, 0.056, -0.001, -0.001]
    gt["target2tt_0"] = euler_vec_to_mat([-90, 0, 90, 1.8, 0, 0.525], use_deg=True)
    return gt 

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

def calibrate_base_to_tt(bag_path=None, perturb=False):
    hand_poses = []
    track_tfs = []
    tt_tfs = []
    pts_all = []
    initials = sim_ground_truth() 

    # read data in either sim mode or actual bag
    if bag_path is not None and os.path.exists(bag_path):
        print(f"Loading bag from {bag_path}")
        read = read_bag
    else:
        print(f"No bag path is given or {bag_path} doesn't exist, running simulation...")
        read = sim_bag_read
    it = read(initials)

    for _, (pts_3d, pts_2d, hand_pose, track_tf, tt_tf) in enumerate(it):
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
    it = read(initials)
    pts_2d_all = [] 
    pts_proj_all = []
    for _, ((pts_3d, pts_2d, _, _, _), track_tf, tt_tf) in enumerate(zip(it, track_tfs, tt_tfs)):
        tf_target2tt_i = tt_tf @ tf_target2tt
        tf_cam2target = np.linalg.inv(tf_target2tt_i) @ tf_track2tt @ track_tf @ tf_base2track @ tf_ee2base @ initials["cam2ee"] 
        pts_proj = transfer_3d_pts_to_img(pts_3d, tf_cam2target, initials["intrinsic"])
        pts_2d_all.append(pts_2d)
        pts_proj_all.append(pts_proj)
    print_reproj_stats(pts_2d_all, pts_proj_all)


if __name__ == "__main__":
    calibrate_base_to_tt(bag_path=None, perturb=True)

