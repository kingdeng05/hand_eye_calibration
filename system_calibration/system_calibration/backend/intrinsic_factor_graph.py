from py_kinetic_backend import Pose3, Cal3Rational, GeneralProjectionFactorCal3Rational 
from py_kinetic_backend import Cal3DS2, GeneralProjectionFactorCal3DS2 
from py_kinetic_backend import Diagonal, NonlinearFactorGraph, symbol
from py_kinetic_backend import Pose3, Values, LevenbergMarquardtOptimizer


def solve_intrinsic_rational(intrinsic_vec, poses, pts_2d, pts_3d):
    graph = NonlinearFactorGraph()
    initial = Values()
    intrinsic_key = symbol('x', 0)
    # initial.insertCal3Rational(intrinsic_key, Cal3Rational(intrinsic_vec))
    initial.insertCal3DS2(intrinsic_key, Cal3DS2(intrinsic_vec))

    pose_keys = []
    proj_noise_model = Diagonal.sigmas([1, 1])
    for idx, pose in enumerate(poses):
        pose_key = symbol('p', idx)
        pose_keys.append(pose_key)
        initial.insertPose3(pose_key, Pose3(pose))

        # general projection factor
        for pt_3d, pt_2d in zip(pts_3d[idx], pts_2d[idx]):
            # graph.add(GeneralProjectionFactorCal3Rational(
            #     pose_key,
            #     intrinsic_key,
            #     pt_3d,
            #     pt_2d,
            #     proj_noise_model,
            #     False,
            #     False 
            # ))
            graph.add(GeneralProjectionFactorCal3DS2(
                pose_key,
                intrinsic_key,
                pt_3d,
                pt_2d,
                proj_noise_model,
                False,
                False 
            ))

    print("graph has ", len(graph), "factors")
    optimizer = LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    # for idx, factor in enumerate(graph):
    #     print(f"{idx} factor error: {factor.error(initial)} => {factor.error(result)}")

    print("error change: {} -> {}".format(graph.error(initial), graph.error(result)))

    return result.atCal3DS2(intrinsic_key).vector(), \
           [result.atPose3(pose_key).matrix() for pose_key in pose_keys] 

        