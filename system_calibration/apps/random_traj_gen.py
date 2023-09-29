import yaml
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from system_calibration.utils import euler_vec_to_mat, mat_to_quaternion 
from system_calibration.simulation import MoveGroup
from system_calibration.simulation.components import ArucoCubeTarget

from build_sim_sys import build_sim_sys, simulate_projection 

np.random.seed(5)

RANGE = np.array([
    np.deg2rad([-60, 60]),
    np.deg2rad([-30, 60]),
    np.deg2rad([-60, 60]),
    [0.4, 0.8],
    [-1, 1],
    [0.5, 1.5],
])

def total_distance(poses):
    distance = 0
    for i in range(len(poses) - 1):
        distance += np.linalg.norm(poses[i, 3:] - poses[i + 1, 3:])
    return distance

def poses_sort(poses):
    occ_ids = set() 
    poses_sort = []
    done = True 
    i = 0
    occ_ids.add(0)
    poses_sort.append(poses[i])
    while done:
        best_idx = -1
        best_pose = None
        largest_distance = np.inf
        for j, pose_j in enumerate(poses):
            if j not in occ_ids:
                dist = np.linalg.norm(np.array(poses[i]) - np.array(pose_j))
                if dist < largest_distance:
                    best_pose = pose_j 
                    largest_distance = dist 
                    best_idx = j
        occ_ids.add(best_idx)
        i = best_idx 
        poses_sort.append(best_pose)
        if len(occ_ids) == len(poses):
            done = False 
    return poses_sort

def quat_to_rot(qx, qy, qz, qw):
    # Convert quaternion to rotation matrix
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def vis_traj(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.2)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.2)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.2)

    for i, pose in enumerate(poses):
        x, y, z, qx, qy, qz, qw = pose
        R = quat_to_rot(qx, qy, qz, qw)

        origin = np.array([x, y, z])

        # Define the axis lengths
        axis_length = 0.1

        # Plot x, y, z-axes
        ax.quiver(origin[0], origin[1], origin[2], R[0, 0], R[1, 0], R[2, 0], color='r', length=axis_length, label="x-axis" if i == "0" else "")
        ax.quiver(origin[0], origin[1], origin[2], R[0, 1], R[1, 1], R[2, 1], color='g', length=axis_length, label="y-axis" if i == "0" else "")
        ax.quiver(origin[0], origin[1], origin[2], R[0, 2], R[1, 2], R[2, 2], color='b', length=axis_length, label="z-axis" if i == "0" else "")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    set_axes_equal(ax)
    plt.show()

def get_random_euler_vec():
    tf_vec = []
    for range in RANGE:
        tf_vec.append(np.random.uniform(*range))
    return tf_vec

def traj_gen_handeye(vis=False):
    # build simulation system
    sim = build_sim_sys()
    pts_cnt_side = len(ArucoCubeTarget(1.035, use_ids=(25,)).get_pts())
    pose_validator = MoveGroup()

    ret = []
    track_val = 0 
    sim.move("track", track_val)
    for _, tt_angle in enumerate(np.linspace(0, 2 * np.pi, 7)[:-1]):
        sim.move("tt", tt_angle) 
        tf = sim.get_transform("cube", "robot")
        yaw = np.arctan2(tf[1, 3], tf[0, 3])
        cnt = 0
        poses = []
        while cnt < 20:
            robot_vec = np.array(get_random_euler_vec()) + np.array([0, 0, yaw, 0, 0, 0]) 
            sim.move("robot", robot_vec)
            try:
                pts_2d, _ = sim.capture("camera", targets=["cube"])
            except RuntimeError as e:
                continue
            if len(pts_2d) < pts_cnt_side:
                continue
            quat = mat_to_quaternion(euler_vec_to_mat(robot_vec)) 
            if pose_validator.is_pose_valid(quat):
                if vis:
                    simulate_projection(pts_2d)
                cnt += 1
                poses.append(quat)
        poses = poses_sort(poses)
        for pose in poses:
            ret.append({
                "tt_angle": tt_angle,
                "track": track_val,
                "robot": pose 
            })
        if vis:
            vis_traj(poses)
    robot_init_pose = [0, 0, 0, 1, 0, 1]
    quat_init = mat_to_quaternion(euler_vec_to_mat(robot_init_pose))
    ret.append({
        "tt_angle": 2 * np.pi,
        "track": 0,
        "robot": quat_init 
    }) 
    return ret

def traj_gen_base2tt(vis=False):
    sim = build_sim_sys()
    robot_pose = [0, 0, 0, 0.6, 0, 0.8]
    sim.move("robot", robot_pose)
    quat = mat_to_quaternion(euler_vec_to_mat(robot_pose))
    ret = []
    for tr in [0.5, 1, 2]: 
        sim.move("track", tr)
        for v in np.linspace(0, 2 * np.pi, 7):
            sim.move("tt", v)
            pts_2d, _ = sim.capture("camera") # project all the targets
            if vis:
                simulate_projection(pts_2d)
            ret.append({
                "tt_angle": v,
                "track": tr,
                "robot": quat 
            })
    return ret

def generate_pose_yaml(ret, name):
    poses_fmt = dict() 
    for i, r in enumerate(ret):
        poses_fmt[i] = {
            'tt_angle': float(r["tt_angle"]),
            'track_position': float(r["track"]),
            'robot_planning_pose': r["robot"] + ["base_link", "link_kinetic", "autel_small_target"]
        }
    with open(name, 'w') as outfile:
        yaml.dump(poses_fmt, outfile)
    
        

if __name__ == "__main__":
    generate_pose_yaml(traj_gen_handeye(vis=False), "hand_eye.yaml") 
    generate_pose_yaml(traj_gen_base2tt(vis=False), "base_tt.yaml") 