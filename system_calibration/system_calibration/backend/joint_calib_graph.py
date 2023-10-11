import numpy as np

from py_kinetic_backend import Cal3Rational, Cam2TtProjectionFactorCal3Rational 
from py_kinetic_backend import Diagonal, NonlinearFactorGraph, symbol
from py_kinetic_backend import Pose3, Values, LevenbergMarquardtOptimizer
from py_kinetic_backend import BaseTtProjectionFactor, PriorFactorPose3
from py_kinetic_backend import BetweenFactorPose3
from py_kinetic_backend import CauchyNoiseModel, RobustNoiseModel 
from py_kinetic_backend import LiDAR2TtPoint2PlaneFactor 

from ..utils import tf_mat_diff

def solve_joint_calib(pts_robot_cam, pts_left_primary, pts_right_primary, track_tfs, tt_tfs, initials):
    # set up the keys
    track2tt_key = symbol('x', 0)
    base2track_key = symbol('x', 1)
    target2tt_key = symbol('x', 2)
    ee2base_key = symbol('x', 3)
    lpc2tt_key = symbol('x', 4)
    rpc2tt_key = symbol('x', 5)
    track_tf_keys = []
    tt_tf_keys = []

    # TODO: debug
    names = []

    # set up noise models
    # proj_noise = Diagonal.sigmas([2, 2]) 
    proj_noise = RobustNoiseModel.create(
        CauchyNoiseModel.create(2.),
        Diagonal.sigmas([2., 2.])
    ) 

    tr2tt_prior_noise = Diagonal.sigmas([1e-3, 1e-3, 1e-6, 0.1, 0.1, 0.1]) # x should be parallel
    base2tr_prior_noise = Diagonal.sigmas([1e-5, 1e-5, 0.1, 1e-4, 1e-4, 1e-4]) # x should be parallel
    hand_prior_noise = Diagonal.sigmas([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]) # from robot manual 
    track_prior_noise = Diagonal.sigmas([1e-8, 1e-8, 1e-8, 1e-3, 1e-8, 1e-8]) # large noise in x 
    tt_prior_noise = Diagonal.sigmas([1e-8, 1e-8, 1e-3, 1e-8, 1e-8, 1e-8]) # should only have yaw
    target2tt_prior_noise = Diagonal.sigmas([1e-5, 1e-2, 1e-5, 1e-1, 1e-4, 1e-1]) # should have mainly roll and y and z are unknown. 
    track_between_noise = Diagonal.sigmas([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]) # should enforce the track state the same

    # set up initial values for time-invariant variables
    values = Values()
    values.insertPose3(track2tt_key, Pose3(initials["track2tt"]))
    values.insertPose3(base2track_key, Pose3(initials["base2track"]))
    values.insertPose3(lpc2tt_key, Pose3(initials["lpc2tt"]))
    values.insertPose3(rpc2tt_key, Pose3(initials["rpc2tt"]))

    graph = NonlinearFactorGraph()

    # assume hand pose doesn't change across time
    values.insertPose3(ee2base_key, Pose3(initials["ee2base"]))
    graph.add(PriorFactorPose3(ee2base_key, Pose3(initials["ee2base"]), hand_prior_noise))
    names.append(f"ee")
    values.insertPose3(target2tt_key, Pose3(initials["target2tt_0"]))
    graph.add(PriorFactorPose3(target2tt_key, Pose3(initials["target2tt_0"]), target2tt_prior_noise))
    names.append(f"target2tt")

    # add calib prior factor
    graph.add(PriorFactorPose3(track2tt_key, Pose3(initials["track2tt"]), tr2tt_prior_noise))
    names.append(f"tr2tt")
    graph.add(PriorFactorPose3(base2track_key, Pose3(initials["base2track"]), base2tr_prior_noise))
    names.append(f"base2t")

    for idx, (pts, pts_lpc, pts_rpc, track_tf, tt_tf) in enumerate(zip(pts_robot_cam, pts_left_primary, pts_right_primary, track_tfs, tt_tfs)):
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
                proj_noise
           ) 
            graph.add(proj_factor)
            names.append(f"proj_{pt_idx}")

        # add left tower camera projection factor
        for pt_idx, (pt_2d, pt_3d) in enumerate(zip(pts_lpc["2d"], pts_lpc["3d"])):
            proj_factor = Cam2TtProjectionFactorCal3Rational(
                lpc2tt_key,
                target2tt_key,
                tt_tf_keys[-1],
                Cal3Rational(initials["intrinsic_lpc"]),
                pt_3d,
                pt_2d,
                proj_noise,
            )
            graph.add(proj_factor)
            names.append(f"lpc_proj_{pt_idx}")

        # add right tower camera projection factor
        for pt_idx, (pt_2d, pt_3d) in enumerate(zip(pts_rpc["2d"], pts_rpc["3d"])):
            proj_factor = Cam2TtProjectionFactorCal3Rational(
                rpc2tt_key,
                target2tt_key,
                tt_tf_keys[-1],
                Cal3Rational(initials["intrinsic_rpc"]),
                pt_3d,
                pt_2d,
                proj_noise,
            )
            graph.add(proj_factor)
            names.append(f"rpc_proj_{pt_idx}")

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

    return {
        "track2tt": result.atPose3(track2tt_key).matrix(),
        "base2track": result.atPose3(base2track_key).matrix(),
        "ee2base": result.atPose3(ee2base_key).matrix(),
        "target2tt_0": result.atPose3(target2tt_key).matrix(),
        "lpc2tt": result.atPose3(lpc2tt_key).matrix(),
        "rpc2tt": result.atPose3(rpc2tt_key).matrix(),
        "track_tfs": [result.atPose3(track_tf_key).matrix() for track_tf_key in track_tf_keys],
        "tt_tfs": [result.atPose3(tt_tf_key).matrix() for tt_tf_key in tt_tf_keys]
    }


def solve_joint_calib_2(pts_robot_cam, pts_left_primary, pts_right_primary, lidar_features, track_tfs, tt_tfs, initials):
    # set up the keys
    track2tt_key = symbol('x', 0)
    base2track_key = symbol('x', 1)
    target2tt_key = symbol('x', 2)
    ee2base_key = symbol('x', 3)
    lpc2tt_key = symbol('x', 4)
    rpc2tt_key = symbol('x', 5)
    lidar2tt_key = symbol('x', 6)
    track_tf_keys = []
    tt_tf_keys = []

    # TODO: debug
    names = []

    # set up noise models
    # proj_noise = Diagonal.sigmas([2, 2]) 
    proj_noise = RobustNoiseModel.create(
        CauchyNoiseModel.create(2.),
        Diagonal.sigmas([2., 2.])
    ) 
    lidar2tt_point2plane_noise = RobustNoiseModel.create(
        CauchyNoiseModel.create(1.),
        Diagonal.sigmas([1e-2]),
    )
    tr2tt_prior_noise = Diagonal.sigmas([1e-3, 1e-3, 1e-6, 0.1, 0.1, 0.1]) # x should be parallel
    base2tr_prior_noise = Diagonal.sigmas([1e-5, 1e-5, 0.1, 1e-4, 1e-4, 1e-4]) # x should be parallel
    hand_prior_noise = Diagonal.sigmas([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]) # from robot manual 
    track_prior_noise = Diagonal.sigmas([1e-8, 1e-8, 1e-8, 1e-3, 1e-8, 1e-8]) # large noise in x 
    tt_prior_noise = Diagonal.sigmas([1e-8, 1e-8, 1e-3, 1e-8, 1e-8, 1e-8]) # should only have yaw
    target2tt_prior_noise = Diagonal.sigmas([1e-5, 1, 1e-5, 1e-1, 1e-4, 1e-1]) # should have mainly roll and y and z are unknown. 
    track_between_noise = Diagonal.sigmas([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]) # should enforce the track state the same

    # set up initial values for time-invariant variables
    values = Values()
    values.insertPose3(track2tt_key, Pose3(initials["track2tt"]))
    values.insertPose3(base2track_key, Pose3(initials["base2track"]))
    values.insertPose3(lpc2tt_key, Pose3(initials["lpc2tt"]))
    values.insertPose3(rpc2tt_key, Pose3(initials["rpc2tt"]))
    values.insertPose3(lidar2tt_key, Pose3(initials["lidar2tt"]))

    graph = NonlinearFactorGraph()

    # assume hand pose doesn't change across time
    values.insertPose3(ee2base_key, Pose3(initials["ee2base"]))
    graph.add(PriorFactorPose3(ee2base_key, Pose3(initials["ee2base"]), hand_prior_noise))
    names.append(f"ee")
    values.insertPose3(target2tt_key, Pose3(initials["target2tt_0"]))
    graph.add(PriorFactorPose3(target2tt_key, Pose3(initials["target2tt_0"]), target2tt_prior_noise))
    names.append(f"target2tt")

    # add calib prior factor
    graph.add(PriorFactorPose3(track2tt_key, Pose3(initials["track2tt"]), tr2tt_prior_noise))
    names.append(f"tr2tt")
    graph.add(PriorFactorPose3(base2track_key, Pose3(initials["base2track"]), base2tr_prior_noise))
    names.append(f"base2t")

    for idx, (pts, pts_lpc, pts_rpc, lidar_feature, track_tf, tt_tf) in enumerate(zip(pts_robot_cam, pts_left_primary, pts_right_primary, lidar_features, track_tfs, tt_tfs)):
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
               names.append(f"track_between-{len(track_tf_keys)-1}-{len(track_tf_keys)}")

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
                proj_noise
           ) 
            graph.add(proj_factor)
            names.append(f"proj_{pt_idx}")

        # add left tower camera projection factor
        for pt_idx, (pt_2d, pt_3d) in enumerate(zip(pts_lpc["2d"], pts_lpc["3d"])):
            proj_factor = Cam2TtProjectionFactorCal3Rational(
                lpc2tt_key,
                target2tt_key,
                tt_tf_keys[-1],
                Cal3Rational(initials["intrinsic_lpc"]),
                pt_3d,
                pt_2d,
                proj_noise,
            )
            graph.add(proj_factor)
            names.append(f"lpc_proj_{pt_idx}")

        # add right tower camera projection factor
        for pt_idx, (pt_2d, pt_3d) in enumerate(zip(pts_rpc["2d"], pts_rpc["3d"])):
            proj_factor = Cam2TtProjectionFactorCal3Rational(
                rpc2tt_key,
                target2tt_key,
                tt_tf_keys[-1],
                Cal3Rational(initials["intrinsic_rpc"]),
                pt_3d,
                pt_2d,
                proj_noise,
            )
            graph.add(proj_factor)
            names.append(f"rpc_proj_{pt_idx}")

        # add lidar feature (pt + surfel) 
        for feature_idx, (pt, surfel) in enumerate(zip(lidar_feature["pts"], lidar_feature["surfels"])):
            lidar_point2plane_factor = LiDAR2TtPoint2PlaneFactor(
                lidar2tt_key,
                target2tt_key,
                tt_tf_keys[-1],
                pt,
                surfel,
                lidar2tt_point2plane_noise,
                False,
                True,
                True 
            )
            graph.add(lidar_point2plane_factor)
            names.append(f"lidar_{feature_idx}")

    optimizer = LevenbergMarquardtOptimizer(graph, values)
    result = optimizer.optimize()

    # analysis
    # for idx, factor in enumerate(graph):
    #     res_error = factor.error(result)
    #     init_error = factor.error(values) 
    #     # if "proj" not in names[idx]:
    #     #     print(f"{names[idx]} factor error: {factor.error(values)} => {factor.error(result)}")
    #     # if res_error > init_error and res_error > 1e-5:
    #     if "lidar" in names[idx]:
    #         print(f"{names[idx]} factor error: {factor.error(values)} => {factor.error(result)}")
    print("error change: {} -> {}".format(graph.error(values), graph.error(result)))

    return {
        "track2tt": result.atPose3(track2tt_key).matrix(),
        "base2track": result.atPose3(base2track_key).matrix(),
        "ee2base": result.atPose3(ee2base_key).matrix(),
        "target2tt_0": result.atPose3(target2tt_key).matrix(),
        "lpc2tt": result.atPose3(lpc2tt_key).matrix(),
        "rpc2tt": result.atPose3(rpc2tt_key).matrix(),
        "lidar2tt": result.atPose3(lidar2tt_key).matrix(),
        "track_tfs": [result.atPose3(track_tf_key).matrix() for track_tf_key in track_tf_keys],
        "tt_tfs": [result.atPose3(tt_tf_key).matrix() for tt_tf_key in tt_tf_keys]
    }

