import numpy as np

from py_kinetic_backend import Diagonal, NonlinearFactorGraph, symbol, Cal3DS2, Cal3Rational
from py_kinetic_backend import Pose3, Values, LevenbergMarquardtOptimizer
from py_kinetic_backend import RMFactorCal3DS2, PriorFactorPose3, PriorFactorCal3DS2
from py_kinetic_backend import HEPoseConstraintFactor, GeneralProjectionFactorCal3DS2, GeneralProjectionFactorCal3Rational 
from py_kinetic_backend import CauchyNoiseModel, RobustNoiseModel 

def calib_rm_factor_graph(calib_init, t2w_init, k_init, hand_poses, pts):
    graph = NonlinearFactorGraph()
    # noise model in pixel
    rm_noise_model = Diagonal.sigmas([2, 2]) 

    # set up symbols 
    calib_key = symbol("x", 0)
    t2w_key = symbol("x", 1)
    k_key = symbol("k", 0)

    # set up initials
    initials = Values()
    initials.insertPose3(calib_key, Pose3(np.linalg.inv(calib_init)))
    initials.insertPose3(t2w_key, Pose3(t2w_init))
    initials.insertCal3DS2(k_key, Cal3DS2(k_init))

    hand_pose_nose_model = Diagonal.sigmas([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4])
    hp_keys = []
    for i, (pose, pts) in enumerate(zip(hand_poses, pts)):
        hp_key = symbol("p", i)
        hp_keys.append(hp_key)
        initials.insertPose3(hp_key, Pose3(np.linalg.inv(pose)))
        pts_2d, pts_3d = pts["2d"], pts["3d"]
        for pt_2d, pt_3d in zip(pts_2d, pts_3d):
            rm_factor = RMFactorCal3DS2(
                calib_key,
                hp_key,
                t2w_key,
                k_key,
                pt_3d,
                pt_2d,
                rm_noise_model,
                False,
                False,
                False,
                True 
            )
            graph.add(rm_factor) 
        # add prior factors for constraining the poses
        graph.add(PriorFactorPose3(hp_key, Pose3(np.linalg.inv(pose)), hand_pose_nose_model))

    # # add prior factors for constraining the poses and intrinsics
    calib_noise_model = Diagonal.sigmas([0.02, 0.02, 0.02, 1e-2, 1e-3, 1e-3])
    graph.add(PriorFactorPose3(calib_key, Pose3(np.linalg.inv(calib_init)), calib_noise_model)) 
    # t2w_noise_model = Diagonal.sigmas([1e-2, 1e-2, 1e-1, 1e-2, 1e-2, 1e-3])
    # graph.add(PriorFactorPose3(t2w_key, Pose3(t2w_init), t2w_noise_model)) 

    optimizer = LevenbergMarquardtOptimizer(graph, initials)
    result = optimizer.optimize()
    print("error change: {} -> {}".format(graph.error(initials), graph.error(result)))

    return np.linalg.inv(result.atPose3(calib_key).matrix()), \
           result.atPose3(t2w_key).matrix(), \
           result.atCal3DS2(k_key).vector(), \
           [np.linalg.inv(result.atPose3(hp_key).matrix()) for hp_key in hp_keys] 
           
def calib_rm2_factor_graph(calib_init, t2w_init, k_init, cam_poses, hand_poses, pts):
    graph = NonlinearFactorGraph()
    # noise model in pixel
    rm_noise_model = Diagonal.sigmas([1., 1.])
    # rm_noise_model = RobustNoiseModel.create(
    #     CauchyNoiseModel.create(3.),
    #     Diagonal.sigmas([1., 1.])
    # ) 

    # set up symbols 
    calib_key = symbol("x", 0)
    w2t_key = symbol("x", 1)
    k_key = symbol("k", 0)

    # set up initials
    initials = Values()
    initials.insertPose3(calib_key, Pose3(calib_init))
    initials.insertPose3(w2t_key, Pose3(np.linalg.inv(t2w_init)))
    initials.insertCal3Rational(k_key, Cal3Rational(k_init))

    hand_meas_noise_model = Diagonal.sigmas([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4])
    t2e_keys = []
    for i, (cam_pose, hand_pose, pts) in enumerate(zip(cam_poses, hand_poses, pts)):
        t2e_key = symbol("p", i)
        t2e_keys.append(t2e_key)
        initials.insertPose3(t2e_key, Pose3(cam_pose))
        pts_2d, pts_3d = pts["2d"], pts["3d"]
        for pt_2d, pt_3d in zip(pts_2d, pts_3d):
            proj_factor = GeneralProjectionFactorCal3Rational(
                t2e_key,
                k_key,
                pt_3d,
                pt_2d,
                rm_noise_model,
                False,
                True
            )
            graph.add(proj_factor)
        he_factor = HEPoseConstraintFactor(
            calib_key,
            w2t_key,
            t2e_key,
            Pose3(hand_pose),
            hand_meas_noise_model,
            False,
            False,
            False
        )
        graph.add(he_factor) 

    calib_noise_model = Diagonal.sigmas([0.02, 0.02, 0.02, 1e-2, 1e-3, 1e-3])
    graph.add(PriorFactorPose3(calib_key, Pose3(calib_init), calib_noise_model)) 
        
    optimizer = LevenbergMarquardtOptimizer(graph, initials)
    result = optimizer.optimize()
    print("error change: {} -> {}".format(graph.error(initials), graph.error(result)))

    return result.atPose3(calib_key).matrix(), \
           np.linalg.inv(result.atPose3(w2t_key).matrix()), \
           result.atCal3Rational(k_key).vector(), \
           [result.atPose3(t2e_key).matrix() for t2e_key in t2e_keys]