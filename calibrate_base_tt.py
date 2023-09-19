import numpy as np
import cv2 as cv
from collections import defaultdict

from py_kinetic_backend import HandPoseFactor, GeneralProjectionFactor, TrackPoseFactor, TtPoseFactor 
from py_kinetic_backend import Diagonal, NonlinearFactorGraph, symbol, Cal3DS2
from py_kinetic_backend import Pose3, Rot3, Values, LevenbergMarquardtOptimizer
from py_kinetic_backend import RMFactorCal3DS2, PriorFactorPose3, PriorFactorCal3DS2

from target import ArucoCubeTarget
from aruco_detector import ArucoDetector
from transform_util import euler_vec_to_mat, mat_to_euler_vec

def solve_base_to_tt_graph(pts_all, hand_poses, track_tfs, tt_tfs, initials):
    # set up the keys
    track2tt_key = symbol('x', 0)
    base2track_key = symbol('x', 1)
    cam2ee_key = symbol('x', 2)
    intrinsic_key = symbol('k', 0)
    target2cam_keys = []
    target2base_keys = []
    target2tt_keys = []

    # set up noise models
    proj_noise = Diagonal.sigmas([2, 2]) 
    hand_noise = Diagonal.sigmas([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]) # from robot manual 
    track_noise = Diagonal.sigmas([1e-5, 1e-5, 1e-5, 1e-2, 1e-4, 1e-4]) # large noise in x 
    tt_noise = Diagonal.sigmas([1e-5, 1e-5, 1e-2, 1e-4, 1e-4, 1e-4]) # should only have yaw
    tr2tt_prior_noise = Diagonal.sigmas([1e-5, 1e-3, 1e-3, 0.1, 0.1, 0.1]) # x should be parallel

    # set up initial values for time-invariant variables
    values = Values()
    values.insert_pose3(track2tt_key, Pose3(initials["track2tt"]))
    values.insert_pose3(base2track_key, Pose3(initials["base2track"]))
    values.insert_pose3(cam2ee_key, Pose3(initials["cam2ee"]))
    values.insert_cal3ds2(intrinsic_key, Cal3DS2(initials["intrinsic"]))
    
    graph = NonlinearFactorGraph()
    for idx, (pts, hand_pose, track_tf, tt_tf) in enumerate(zip(pts_all, hand_poses, track_tfs, tt_tfs)):
        # add keys
        target2cam_keys.apend(symbol('c', idx))
        target2base_keys.apend(symbol('b', idx))
        target2tt_keys.apend(symbol('t', idx))

        # insert values
        values.insert_pose3(target2base_keys[-1], initials["target2base"][idx])
        values.insert_pose3(target2cam_keys[-1], initials["target2cam"][idx])
        values.insert_pose3(target2tt_keys[-1], initials["target2tt"][idx])

        # add projection factor
        for pt_3d, pt_2d in zip(pts["3d"], pts["2d"]):
            proj_factor = GeneralProjectionFactor(target2cam_keys[-1], intrinsic_key, \
                                                  pt_3d, pt_2d, proj_noise, \
                                                  False, True)
            graph.add(proj_factor)

        # add hand pose factor
        hand_factor = HandPoseFactor(cam2ee_key, target2base_keys[-1], target2cam_keys[-1], \
                                     Pose3(hand_pose), hand_noise, True, False, False)
        graph.add(hand_factor)

        # add track pose factor
        track_factor = TrackPoseFactor(base2track_key, target2base_keys[-1], target2tt_keys[-1], \
                                       track2tt_key, Pose3(track_tf), track_noise, \
                                       False, False, False, False) 
        graph.add(track_factor)

        # add tt pose factor
        tt_factor = TtPoseFactor(target2tt_keys[-1], target2tt_keys[0], Pose3(tt_tf), tt_noise)
        graph.add(tt_factor)

        # add tr2tt prior factor
        tr2tt_prior_factor = PriorFactorPose3(track2tt_key, Pose3(initials["track2tt"]), tr2tt_prior_noise)
        graph.add(tr2tt_prior_factor)
    
    optimizer = LevenbergMarquardtOptimizer(graph, initials)
    result = optimizer.optimize()
    print("error change: {} -> {}".format(graph.error(initials), graph.error(result)))

    return result.atPose3(track2tt_key).matrix(), \
           result.atPose3(base2track_key).matrix(), \
           [result.atPose3(target2tt_key).matrix() for target2tt_key in target2tt_keys], \
           [result.atPose3(target2base_key).matrix() for target2base_key in target2base_keys], \
           [result.atPose3(target2cam_key).matrix() for target2cam_key in target2cam_keys]

def sim_bag_read():
    pass

def sim_initials():
    pass

def track_reading_to_transform(track_reading):
    # track moves in x direction, 0 is at the very begining
    # so only negative values
    tf = np.eye(4)
    tf[0, 3] = track_reading 
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

def calibrate_base_to_tt_sim():
    calib_target = ArucoCubeTarget(1.035)
    aruco_detector = ArucoDetector()

    hand_poses = []
    track_tfs = []
    tt_tfs = []
    pts_all = []
    for _, (img, hand_pose, track_reading, tt_reading) in enumerate(sim_bag_read()):
        track_tfs.append(track_reading_to_transform(track_reading))
        tt_tfs.append(tt_reading_to_transform(tt_reading))
        hand_poses.append(hand_pose) 
        ret = aruco_detector.detect(img)
        pts_pair = defaultdict(list) 
        for id, corner in zip(*ret):
            pts_3d = calib_target.find_3d_pts_by_id(id)
            if pts_3d is not None:
                pts_pair["2d"].append(corner)
                pts_pair["3d"].append(pts_3d)
        pts_all.append(pts_pair)

    initials = sim_initials()
    tf_track2tt, tf_base2track, tfs_target2tt, tfs_target2base, tfs_target2cam = solve_base_to_tt_graph(pts_all, hand_poses, track_tfs, tt_tfs, initials)
    print(tf_track2tt)
    print(tf_base2track)
                

