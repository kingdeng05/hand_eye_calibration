import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from system_calibration.utils import euler_vec_to_mat, mat_to_quaternion 
from system_calibration.simulation import MoveGroup
from system_calibration.simulation.components import ArucoCubeTarget

from build_sim_sys import build_sim_sys, simulate_projection 

np.random.seed(5)

RANGE = np.array([
    np.deg2rad([-30, 30]),
    np.deg2rad([-30, 30]),
    np.deg2rad([-30, 30]),
    [0.1, 0.5],
    [-1, 1],
    [0.2, 0.6],
])


def get_random_euler_vec():
    tf_vec = []
    for range in RANGE:
        tf_vec.append(np.random.uniform(*range))
    return tf_vec

def traj_gen():
    # build simulation system
    sim = build_sim_sys()
    pts_cnt_side = len(ArucoCubeTarget(1.035, use_ids=(25,)).get_pts())
    pose_validator = MoveGroup()

    ret = []
    track_val = 0 
    sim.move("track", track_val)
    for idx, tt_angle in enumerate(np.linspace(0, 2 * np.pi, 6)):
        sim.move("tt", tt_angle) 
        tf = sim.get_transform("cube", "robot")
        print("finished")
        print(tf[:3, 3])
        yaw = np.arctan2(tf[1, 3], tf[0, 3])
        cnt = 0
        print("angle: ", idx, np.rad2deg(yaw))
        while cnt < 20:
            robot_vec = np.array(get_random_euler_vec()) + np.array([0, 0, yaw, 0, 0, 0]) 
            sim.move("robot", robot_vec)
            try:
                pts_2d, _ = sim.capture_camera("camera")
            except RuntimeError:
                continue
            if len(pts_2d) < pts_cnt_side:
                continue
            quat = mat_to_quaternion(euler_vec_to_mat(robot_vec)) 
            if pose_validator.is_pose_valid(quat):
                ret.append({
                    "tt_angle": tt_angle,
                    "track": track_val,
                    "robot": quat
                }) 
                # print(quat)
                simulate_projection(pts_2d)
                cnt += 1
        

if __name__ == "__main__":
    traj_gen()