import numpy as np

from py_kinetic_backend import HandPoseFactor, TrackPoseFactor, TtPoseFactor 
from py_kinetic_backend import Cal3Rational, GeneralProjectionFactorCal3Rational 
from py_kinetic_backend import Diagonal, NonlinearFactorGraph, symbol
from py_kinetic_backend import Pose3, Values, LevenbergMarquardtOptimizer
from py_kinetic_backend import BaseTtProjectionFactor, PriorFactorPose3
from py_kinetic_backend import BetweenFactorPose3 

from ..utils import tf_mat_diff

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
    target2tt_prior_noise = Diagonal.sigmas([1e-2, 1e-5, 1e-5, 1e-1, 1e-4, 1e-1]) # should have mainly roll and y and z are unknown. 
    track_between_noise = Diagonal.sigmas([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]) # should enforce the track state the same

    # set up initial values for time-invariant variables
    values = Values()
    values.insertPose3(track2tt_key, Pose3(initials["track2tt"]))
    values.insertPose3(base2track_key, Pose3(initials["base2track"]))

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

        # these constraints need to be added
        # because track should be the static across multiple
        # turntable readings 
        if len(track_tf_keys) >= 2:
            diff_vec = tf_mat_diff(
                values.atPose3(track_tf_keys[-2]).matrix(),
                values.atPose3(track_tf_keys[-1]).matrix()
            )
            # this is not elegant but will do the trick now
            if abs(diff_vec[3]) < 0.01:
               between_factor = BetweenFactorPose3(
                   track_tf_keys[-2],
                   track_tf_keys[-1],
                   Pose3(np.eye(4)),
                   track_between_noise
               ) 
               graph.add(between_factor)

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
                Cal3Rational(initials["intrinsic"]),
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
           [result.atPose3(tt_tf_key).matrix() for tt_tf_key in tt_tf_keys]


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
    # track2tt_pert = Pose3(perturb_pose3(initials["track2tt"], 0.01 * np.ones(6)))
    # base2track_pert = Pose3(perturb_pose3(initials["base2track"], 0.01 * np.ones(6)))
    # print("initial diff track2tt: ", mat_to_euler_vec(np.linalg.inv(initials["track2tt"]) @ track2tt_pert.matrix()))
    # print("initial diff base2track: ", mat_to_euler_vec(np.linalg.inv(initials["base2track"]) @ base2track_pert.matrix()))

    # set up initial values for time-invariant variables
    values = Values()
    values.insertPose3(track2tt_key, Pose3(initials["track2tt"]))
    values.insertPose3(base2track_key, Pose3(initials["base2track"]))
    values.insertPose3(cam2ee_key, Pose3(initials["cam2ee"]))
    values.insertCal3Rational(intrinsic_key, Cal3Rational(initials["intrinsic"]))

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
            proj_factor = GeneralProjectionFactorCal3Rational(target2cam_keys[-1], intrinsic_key, \
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


