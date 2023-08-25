import numpy as np

from py_kinetic_backend import Diagonal, NonlinearFactorGraph, symbol, Pose3, Rot3
from py_kinetic_backend import DHEFactor, Values, LevenbergMarquardtOptimizer


def calibrate_dhe_factor_graph(calib_gt, poses_a, poses_b):
    # Create a noise model for the measurements
    dhe_noise_model = Diagonal.sigmas(np.append(np.deg2rad([0.1, 0.1, 0.1]), 0.1 * np.ones(3)))

    # Initialize factor graph
    graph = NonlinearFactorGraph()

    calib_key = symbol('X', 0)
    initial = Values()

    # Add the first prior factors
    prior_pose = Pose3(poses_a[0])
    initial.insertPose3(symbol('E', 0), prior_pose)

    prior_pose = Pose3(poses_b[0])
    initial.insertPose3(symbol('C', 0), prior_pose)

    for idx in range(1, len(poses_a)):
        prior_pose = Pose3(poses_a[idx])
        initial.insertPose3(symbol('E', idx), prior_pose)

        prior_pose = Pose3(poses_b[idx])
        initial.insertPose3(symbol('C', idx), prior_pose)

        dhe_factor = DHEFactor(calib_key, symbol('E', idx-1), symbol('E', idx), 
                               symbol('C', idx-1), symbol('C', idx), dhe_noise_model)
        graph.add(dhe_factor)

    # Create a small perturbation
    rot = Rot3.RzRyRx([0.01, 0.02, 0.01])
    pert = Pose3(rot, np.array([0.02, 0.04, 0.03]))

    calib_gt_pose = Pose3(calib_gt)
    calib_gt_pose = calib_gt_pose.compose(pert)
    initial.insertPose3(calib_key, calib_gt_pose)

    # Optimize the factor graph
    optimizer = LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    print("error change: {} -> {}".format(graph.error(initial), graph.error(result)))

    vec_diff = Pose3.logmap(Pose3(calib_gt).between(result.atPose3(calib_key)))
    print("rot diff: {}".format(np.rad2deg(vec_diff[:3])))
    print("t diff: {}".format(vec_diff[3:]))
