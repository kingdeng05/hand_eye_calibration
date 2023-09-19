import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from py_kinetic_backend import Pose3, Rot3, Cal3DS2, PinholeCameraCal3DS2

from sim import * 
from target import CheckerBoardTarget, ArucoCubeTarget
from robot_pose_validator import MoveGroup

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


def tf_matrix_to_quaternion(pose):
    quat = Rot3(pose[:3, :3]).quaternion()  # qw, qx, qy, qz
    return pose[:3, 3].tolist() +  quat[[1, 2, 3, 0]].tolist()

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
    # this needs to have actual measurement
    intrinsic = create_intrinsic_distortion(focal_length="4mm")
    hand_tf_camera = create_calib_gt()
    base_tf_target = create_cube_t2w_gt()
    pts_target_3d = ArucoCubeTarget(1.035, use_ids=(50, 100)).get_pts_3d()
    visible_cam = VisibleCamera(intrinsic)
    mg = MoveGroup()

    # start generating
    valid_cnt = 0
    hand_poses = []
    while valid_cnt < 100:
        target_tf_camera = get_random_transformation()
        # perform projection
        camera = PinholeCameraCal3DS2(Pose3(target_tf_camera), Cal3DS2(intrinsic))
        pts_2d = []
        for pt_3d in pts_target_3d:
            try:
                pt_2d = camera.project(pt_3d)         
            except:
                continue
            if pt_2d[0] < 0 or pt_2d[0] >= IMG_WIDTH or pt_2d[1] < 0 or pt_2d[1] >= IMG_HEIGHT: 
                continue 
            pts_2d.append(pt_2d)
        if len(pts_2d) / len(pts_target_3d) < 0.5:
            continue
        base_tf_hand = base_tf_target @ target_tf_camera @ np.linalg.inv(hand_tf_camera) 
        # if base_tf_hand[0, 3] > 0.03 and base_tf_hand[2, 3] > 0.1 and np.linalg.norm(base_tf_hand[:3, 3]) < 1.2:
        quat = tf_matrix_to_quaternion(base_tf_hand)
        if mg.is_pose_valid(quat):
            valid_cnt += 1
            hand_poses.append(quat)
            print("simulated pose: ", valid_cnt)
            # vis
            if VIS:
                raw_img = np.zeros((IMG_HEIGHT, IMG_WIDTH))
                pts_2d = visible_cam.project(target_tf_camera, ArucoCubeTarget(1.035, use_ids=(0, 25, 50, 75, 100)), raw_img.shape[::-1]).astype(int)
                print(len(pts_2d))
                # pts_2d = np.array(pts_2d).astype(int)
                for pt_2d in pts_2d:
                    cv.circle(raw_img, tuple(pt_2d), 2, 255) 
                raw_img = cv.resize(raw_img, (int(IMG_WIDTH/2), int(IMG_HEIGHT/2)))
                # cv.imwrite(f"simulated-img{valid_cnt}.png", raw_img)
                cv.imshow("target_projection", raw_img)
                cv.waitKey(0)

    return hand_poses

if __name__ == "__main__":
    traj_gen()