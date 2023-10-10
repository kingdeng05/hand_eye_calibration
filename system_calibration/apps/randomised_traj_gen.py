import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from py_kinetic_backend import Pose3, Rot3

from system_calibration.simulation.components import ArucoCubeTarget 
from system_calibration.simulation import MoveGroup
from system_calibration.utils import draw_pts_on_img, mat_to_euler_vec 
from system_calibration.utils import mat_to_quaternion 

from build_sim_sys import build_sim_sys, get_img_size

VIS = True 
np.random.seed(5)

RANGE = np.array([
    np.deg2rad([-60, 60]),
    np.deg2rad([-60, 60]),
    np.deg2rad([-60, 60]),
    [-0.5, 0.5],
    [-0.2, 0.2],
    [-3, -2],
])


def plot_frame(ax, transformation_matrix, name):
    origin = transformation_matrix[:3, 3]
    orientation = transformation_matrix[:3, :3]
    
    axis_length = 1 
    
    # Plot each axis of the frame
    for j in range(3):
        end_point = origin + orientation[:, j] * axis_length
        ax.quiver(origin[0], origin[1], origin[2], end_point[0] - origin[0],
                  end_point[1] - origin[1], end_point[2] - origin[2], color=['r', 'g', 'b'][j], arrow_length_ratio=0.1)
    
    # Annotate the frame with the given name
    ax.text(origin[0], origin[1], origin[2], name)

def get_random_transformation():
    tf_vec = []
    for range in RANGE:
        tf_vec.append(np.random.uniform(*range))
    tf_vec = Pose3(Rot3.rzryrx(tf_vec[:3]), tf_vec[3:]).matrix()
    return tf_vec

def traj_gen():
    sim = build_sim_sys()
    # this needs to have actual measurement
    hand_tf_camera = sim.get_transform("camera", "robot", to_base=False) 
    base_tf_target = sim.get_transform("cube", "robot", to_base=True) 
    mg = MoveGroup()

    # start generating
    valid_cnt = 0
    hand_poses = []
    while valid_cnt < 100:
        target_tf_camera = get_random_transformation()
        base_tf_hand = base_tf_target @ target_tf_camera @ np.linalg.inv(hand_tf_camera) 
        sim.move("robot", mat_to_euler_vec(base_tf_hand, use_deg=False))
        try:
            pts_2d, _ = sim.capture("camera")
        except:
            continue
        if len(pts_2d) < 30:
            print("skipped")
            continue
        quat = mat_to_quaternion(base_tf_hand)
        if mg.is_pose_valid(quat):
            valid_cnt += 1
            hand_poses.append(quat)
            print("simulated pose: ", valid_cnt)
            # vis
            if VIS:
                width, height = get_img_size()
                raw_img = np.zeros((height, width, 3))
                raw_img = draw_pts_on_img(raw_img, pts_2d) 
                raw_img = cv.resize(raw_img, (int(width/2), int(height/2)))
                # cv.imwrite(f"simulated-img{valid_cnt}.png", raw_img)
                cv.imshow("target_projection", raw_img)
                cv.waitKey(0)

    return hand_poses

if __name__ == "__main__":
    traj_gen()