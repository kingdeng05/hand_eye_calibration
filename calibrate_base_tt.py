import numpy as np
import cv2 as cv
from copy import deepcopy 

from py_kinetic_backend import HandPoseFactor, GeneralProjectionFactorCal3DS2, TrackPoseFactor, TtPoseFactor 
from py_kinetic_backend import Diagonal, NonlinearFactorGraph, symbol, Cal3DS2
from py_kinetic_backend import Pose3, Values, LevenbergMarquardtOptimizer
from py_kinetic_backend import BaseTtProjectionFactor, PriorFactorPose3

from sim import *
from target import ArucoCubeTarget
from aruco_detector import ArucoDetector
from transform_util import euler_vec_to_mat, mat_to_euler_vec, transfer_3d_pts_to_img
from calibrate_intrinsic import calculate_reproj_error

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

def solve_base_to_tt_graph(pts_all, hand_poses, track_tfs, tt_tfs, initials):
    # set up the keys
    track2tt_key = symbol('x', 0)
    base2track_key = symbol('x', 1)
    cam2ee_key = symbol('x', 2)
    intrinsic_key = symbol('k', 0)
    target2cam_keys = []
    target2base_keys = []
    target2tt_keys = []

    # TODO: debug
    names = []

    # set up noise models
    proj_noise = Diagonal.sigmas([2, 2]) 
    hand_noise = Diagonal.sigmas([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]) # from robot manual 
    track_noise = Diagonal.sigmas([1e-5, 1e-5, 1e-5, 1e-2, 1e-4, 1e-4]) # large noise in x 
    tt_noise = Diagonal.sigmas([1e-8, 1e-8, 2e-3, 1e-4, 1e-4, 1e-4]) # should only have yaw
    tr2tt_prior_noise = Diagonal.sigmas([1e-3, 1e-3, 1e-6, 0.1, 0.1, 0.1]) # x should be parallel
    base2tr_prior_noise = Diagonal.sigmas([1e-3, 1e-3, 0.1, 1e-4, 1e-4, 1e-4]) # x should be parallel

    # perform noise injection
    # track2tt_pert = Pose3(initials["track2tt"])
    # base2track_pert = Pose3(initials["base2track"])
    track2tt_pert = Pose3(perturb_pose3(initials["track2tt"], 0.01 * np.ones(6)))
    base2track_pert = Pose3(perturb_pose3(initials["base2track"], 0.01 * np.ones(6)))
    print("initial diff track2tt: ", mat_to_euler_vec(np.linalg.inv(initials["track2tt"]) @ track2tt_pert.matrix()))
    print("initial diff base2track: ", mat_to_euler_vec(np.linalg.inv(initials["base2track"]) @ base2track_pert.matrix()))

    # set up initial values for time-invariant variables
    values = Values()
    values.insertPose3(track2tt_key, track2tt_pert)
    values.insertPose3(base2track_key, base2track_pert)
    values.insertPose3(cam2ee_key, Pose3(initials["cam2ee"]))
    values.insertCal3DS2(intrinsic_key, Cal3DS2(initials["intrinsic"]))

    graph = NonlinearFactorGraph()
    for idx, (pts, hand_pose, track_tf, tt_tf) in enumerate(zip(pts_all, hand_poses, track_tfs, tt_tfs)):
        # add keys
        target2cam_keys.append(symbol('c', idx))
        target2base_keys.append(symbol('b', idx))
        target2tt_keys.append(symbol('t', idx))

        # insert values
        values.insertPose3(target2base_keys[-1], Pose3(initials["target2base"][idx]))
        values.insertPose3(target2cam_keys[-1], Pose3(initials["target2cam"][idx]))
        values.insertPose3(target2tt_keys[-1], Pose3(initials["target2tt"][idx]))

        # add projection factor
        for pt_idx, (pt_3d, pt_2d) in enumerate(zip(pts["3d"], pts["2d"])):
            proj_factor = GeneralProjectionFactorCal3DS2(target2cam_keys[-1], intrinsic_key, \
                                                         pt_3d, pt_2d, proj_noise, \
                                                         False, True)
            graph.add(proj_factor)
            names.append(f"proj_{pt_idx}")

        # add hand pose factor
        hand_factor = HandPoseFactor(cam2ee_key, target2base_keys[-1], target2cam_keys[-1], \
                                     Pose3(hand_pose), hand_noise, True, False, False)
        graph.add(hand_factor)
        names.append(f"hand_{idx}")

        # add track pose factor
        track_factor = TrackPoseFactor(base2track_key, target2base_keys[-1], target2tt_keys[-1], \
                                       track2tt_key, Pose3(track_tf), track_noise, \
                                       False, False, False, False) 
        graph.add(track_factor)
        names.append(f"track_{idx}")

        # add tt pose factor
        tt_factor = TtPoseFactor(target2tt_keys[-1], target2tt_keys[0], Pose3(tt_tf), tt_noise)
        graph.add(tt_factor)
        names.append(f"tt_{idx}")

        # add tr2tt prior factor
        tr2tt_prior_factor = PriorFactorPose3(track2tt_key, Pose3(initials["track2tt"]), tr2tt_prior_noise)
        graph.add(tr2tt_prior_factor)
        names.append(f"tr2tt_{idx}")
    
        # add base2tr prior factor
        base2tr_prior_factor = PriorFactorPose3(base2track_key, Pose3(initials["base2track"]), base2tr_prior_noise)
        graph.add(base2tr_prior_factor)
        names.append(f"base2tr_{idx}")

    optimizer = LevenbergMarquardtOptimizer(graph, values)
    result = optimizer.optimize()
    print("error change: {} -> {}".format(graph.error(values), graph.error(result)))

    # analysis
    for idx, factor in enumerate(graph):
        res_error = factor.error(result)
        init_error = factor.error(values) 
        if res_error > init_error and res_error > 1e-5:
            print(f"{names[idx]} factor error: {factor.error(values)} => {factor.error(result)}")

    return result.atPose3(track2tt_key).matrix(), \
           result.atPose3(base2track_key).matrix(), \
           [result.atPose3(target2tt_key).matrix() for target2tt_key in target2tt_keys], \
           [result.atPose3(target2base_key).matrix() for target2base_key in target2base_keys], \
           [result.atPose3(target2cam_key).matrix() for target2cam_key in target2cam_keys]

def solve_base_to_tt_graph_2(pts_all, hand_poses, track_tfs, tt_tfs, initials):
    # set up the keys
    track2tt_key = symbol('x', 0)
    base2track_key = symbol('x', 1)
    target2tt_key = symbol('x', 3)
    ee2base_key = symbol('x', 4)
    track_tf_keys = []
    tt_tf_keys = []

    # TODO: debug
    names = []

    # set up noise models
    proj_noise = Diagonal.sigmas([1, 1]) 
    tr2tt_prior_noise = Diagonal.sigmas([1e-3, 1e-3, 1e-6, 0.1, 0.1, 0.1]) # x should be parallel
    base2tr_prior_noise = Diagonal.sigmas([1e-5, 1e-5, 0.1, 1e-4, 1e-4, 1e-4]) # x should be parallel
    hand_prior_noise = Diagonal.sigmas([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]) # from robot manual 
    track_prior_noise = Diagonal.sigmas([1e-8, 1e-8, 1e-8, 1e-3, 1e-8, 1e-8]) # large noise in x 
    tt_prior_noise = Diagonal.sigmas([1e-8, 1e-8, 1e-3, 1e-8, 1e-8, 1e-8]) # should only have yaw
    target2tt_prior_noise = Diagonal.sigmas([1e-2, 1e-5, 1e-5, 1e-4, 1e-1, 1e-1]) # should have mainly roll and y and z are unknown. 

    # perform noise injection
    # track2tt_pert = Pose3(initials["track2tt"])
    # base2track_pert = Pose3(initials["base2track"])
    track2tt_pert = Pose3(perturb_pose3(initials["track2tt"], 0.1 * np.ones(6)))
    base2track_pert = Pose3(perturb_pose3(initials["base2track"], 0.1 * np.ones(6)))
    print("initial diff track2tt: ", mat_to_euler_vec(np.linalg.inv(initials["track2tt"]) @ track2tt_pert.matrix()))
    print("initial diff base2track: ", mat_to_euler_vec(np.linalg.inv(initials["base2track"]) @ base2track_pert.matrix()))

    # set up initial values for time-invariant variables
    values = Values()
    values.insertPose3(track2tt_key, track2tt_pert)
    values.insertPose3(base2track_key, base2track_pert)

    graph = NonlinearFactorGraph()

    # assume hand pose doesn't change across time
    values.insertPose3(ee2base_key, Pose3(hand_poses[0]))
    graph.add(PriorFactorPose3(ee2base_key, Pose3(hand_poses[0]), hand_prior_noise))
    names.append(f"ee")
    values.insertPose3(target2tt_key, Pose3(initials["target2tt_0"]))
    graph.add(PriorFactorPose3(target2tt_key, Pose3(initials["target2tt_0"]), target2tt_prior_noise))
    names.append(f"target2tt")

    # add calib prior factor
    graph.add(PriorFactorPose3(track2tt_key, Pose3(initials["track2tt"]), tr2tt_prior_noise))
    names.append(f"tr2tt")
    graph.add(PriorFactorPose3(base2track_key, Pose3(initials["base2track"]), base2tr_prior_noise))
    names.append(f"base2t")

    for idx, (pts, _, track_tf, tt_tf) in enumerate(zip(pts_all, hand_poses, track_tfs, tt_tfs)):
        # add keys
        track_tf_keys.append(symbol('a', idx))
        tt_tf_keys.append(symbol('b', idx))

        # insert values
        values.insertPose3(track_tf_keys[-1], Pose3(track_tf))
        values.insertPose3(tt_tf_keys[-1], Pose3(tt_tf))

        # add track and tt prior factor
        graph.add(PriorFactorPose3(track_tf_keys[-1], Pose3(track_tf), track_prior_noise))
        names.append(f"track_{idx}")
        graph.add(PriorFactorPose3(tt_tf_keys[-1], Pose3(tt_tf), tt_prior_noise))
        names.append(f"tt_{idx}")

        # add projection factor
        for pt_idx, (pt_3d, pt_2d) in enumerate(zip(pts["3d"], pts["2d"])):
            proj_factor = BaseTtProjectionFactor(
                ee2base_key,
                base2track_key,
                track_tf_keys[-1],
                track2tt_key,
                tt_tf_keys[-1],
                target2tt_key,
                pt_3d,
                pt_2d,
                Pose3(initials["cam2ee"]),
                Cal3DS2(initials["intrinsic"]),
                proj_noise,
                False,
                False,
                False,
                False,
                False,
                False
            ) 
            graph.add(proj_factor)
            names.append(f"proj_{pt_idx}")

    optimizer = LevenbergMarquardtOptimizer(graph, values)
    result = optimizer.optimize()

    # analysis
    # for idx, factor in enumerate(graph):
    #     res_error = factor.error(result)
    #     init_error = factor.error(values) 
    #     # if "proj" not in names[idx]:
    #     #     print(f"{names[idx]} factor error: {factor.error(values)} => {factor.error(result)}")
    #     if res_error > init_error and res_error > 1e-5:
    #         print(f"{names[idx]} factor error: {factor.error(values)} => {factor.error(result)}")
    print("error change: {} -> {}".format(graph.error(values), graph.error(result)))

    return result.atPose3(track2tt_key).matrix(), \
           result.atPose3(base2track_key).matrix(), \
           result.atPose3(ee2base_key).matrix(), \
           result.atPose3(target2tt_key).matrix(), \
           [result.atPose3(track_tf_key).matrix() for track_tf_key in track_tf_keys], \
           [result.atPose3(tt_tf_key).matrix() for tt_tf_key in tt_tf_keys], \


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

def calibrate_base_to_tt(sim=True, calib_bag_path=''):
    hand_poses = []
    track_tfs = []
    tt_tfs = []
    pts_all = []
    initials = sim_ground_truth() 

    # read data in either sim mode or actual bag
    if sim:
        it = sim_bag_read(initials)
    else:
        it = read_bag(initials)

    for _, (pts_3d, pts_2d, hand_pose, track_tf, tt_tf) in enumerate(it):
        track_tfs.append(track_tf)
        tt_tfs.append(tt_tf)
        tf_target2tt_i = tt_tf @ initials["target2tt_0"] 
        tf_target2base_i = np.linalg.inv(initials["track2tt"] @ track_tf @ initials["base2track"]) @ tf_target2tt_i
        tf_target2cam_i = np.linalg.inv(hand_pose @ initials["cam2ee"]) @ tf_target2base_i
        append_dict_list(initials, "target2tt", tf_target2tt_i) 
        append_dict_list(initials, "target2cam", tf_target2cam_i) 
        append_dict_list(initials, "target2base", tf_target2base_i) 
        hand_poses.append(hand_pose) 
        pts_pair = dict()
        pts_pair["3d"] = pts_3d
        pts_pair["2d"] = pts_2d
        pts_all.append(pts_pair)

    # add perturbation 
    pert = deepcopy(initials)
    pert["track2tt"] = perturb_pose3(pert["track2tt"], [1e-3, 1e-3, 0, 0.1, 0.1, 0.1]) 
    pert["base2track"] = perturb_pose3(pert["base2track"], [1e-4, 1e-4, 0.1, 0, 0, 0]) 
    track_tfs = [perturb_pose3(tr_tf, [0, 0, 0, 1e-3, 0, 0]) for tr_tf in track_tfs]
    tt_tfs = [perturb_pose3(tt_tf, [0, 0, 1e-4, 0, 0, 0]) for tt_tf in tt_tfs]
    # hand_poses = [perturb_pose3(hand_pose, [1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]) for hand_pose in hand_poses]
    # pts_all = [{"2d": perturb_pts(pts["2d"], [1, 1]), "3d": pts["3d"]} for pts in pts_all] 

    print("frontend features collection finished, running backend...")
    tf_track2tt, tf_base2track, tfs_target2tt, tfs_target2base, tfs_target2cam = solve_base_to_tt_graph(pts_all, hand_poses, track_tfs, tt_tfs, initials)
    print("diff under track coord: ", mat_to_euler_vec(np.linalg.inv(initials["track2tt"]) @ tf_track2tt))
    print("diff under base coord: ", mat_to_euler_vec(np.linalg.inv(initials["target2tt_0"]) @ tf_base2track))

def calibrate_base_to_tt_2(sim=True, calib_bag_path='', perturb=False):
    hand_poses = []
    track_tfs = []
    tt_tfs = []
    pts_all = []
    initials = sim_ground_truth() 

    # read data in either sim mode or actual bag
    read = sim_bag_read if sim else read_bag 
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
    # calibrate_base_to_tt(sim=True)
    calibrate_base_to_tt_2(sim=True, perturb=True)

