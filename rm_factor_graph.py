import numpy as np

from py_kinetic_backend import Diagonal, NonlinearFactorGraph, symbol, Cal3_S2
from py_kinetic_backend import RMFactorCal3_S2, RMFactorCal3DS2, Values, LevenbergMarquardtOptimizer
from py_kinetic_backend import Pose3 

def calib_rm_factor_graph(calib_init, t2w_init, k_init, hand_poses, pts):
    graph = NonlinearFactorGraph()
    # noise model in pixel
    rm_noise_model = Diagonal.sigmas(np.deg2rad([0.1, 0.1])) 

    initials = Values()
    calib_key = symbol("x", 0)
    t2w_key = symbol("x", 1)
    k_key = symbol("k", 0)
    initials.insertPose3(calib_key, Pose3(np.linalg.inv(calib_init)))
    initials.insertPose3(t2w_key, Pose3(t2w_init))
    initials.insertCal3_S2(k_key, Cal3_S2(k_init))

    for i, (pose, pts) in enumerate(zip(hand_poses, pts)):
        hp_key = symbol("p", i)
        initials.insertPose3(hp_key, Pose3(np.linalg.inv(pose)))
        pts_2d, pts_3d = pts["2d"], pts["3d"]
        for pt_2d, pt_3d in zip(pts_2d, pts_3d):
            rm_factor = RMFactorCal3_S2(
                calib_key,
                hp_key,
                t2w_key,
                k_key,
                pt_3d,
                pt_2d,
                rm_noise_model,
                False,
                True,
                False,
                False 
            )
            graph.add(rm_factor) 

    optimizer = LevenbergMarquardtOptimizer(graph, initials)
    result = optimizer.optimize()
    print("error change: {} -> {}".format(graph.error(initials), graph.error(result)))

    return np.linalg.inv(result.atPose3(calib_key).matrix()), \
           result.atPose3(t2w_key).matrix()
           
