import yaml
import numpy as np
import cv2 as cv
from scipy.spatial import KDTree

from system_calibration.IO import TopicTriggerBagReader
from system_calibration.utils import msg_to_img, pose_msg_to_tf, quaternion_to_mat, euler_vec_to_mat
from system_calibration.utils import pc2_msg_to_array, plane_fitting, numpy_to_pcd, vis_points_with_normal 
from system_calibration.utils import transform_3d_pts, mat_to_euler_vec 

from build_sim_sys import build_sim_sys


def read_intrinsic_bag(bag_name):
    topics = [
        # "/robot_0/robot_base/end_effector_pose_stopped",
        "/tt/system_stopped",
        # "/camera/image_color/compressed",
        "/camera/image_raw/compressed",
    ]
    reader = TopicTriggerBagReader(bag_name, *topics)
    for msgs in reader.read():
        img = msg_to_img(msgs[1][1], RGB=False)
        yield img

def read_hand_eye_bag(bag_name):
    topics = [
        # "/robot_0/robot_base/end_effector_pose_stopped",
        "/tt/system_stopped",
        "/camera/image_raw/compressed",
        "/robot_0/robot_base/end_effector_pose",
    ]
    reader = TopicTriggerBagReader(bag_name, *topics)
    for idx, msgs in enumerate(reader.read()):
        # only the first 30 frames are in the same target
        # if idx < 30:
        if idx >= 30:
            yield msg_to_img(msgs[1][1], RGB=False), pose_msg_to_tf(msgs[2][1].pose) 

# TODO: this is get the robot pose from pose yaml
def read_hand_eye_bag_adhoc(bag_name, pose_yaml):
    topics = [
        "/robot_0/robot_base/end_effector_pose_stopped",
        "/camera/image_color/compressed",
    ]
    pose_cfg = yaml.safe_load(open(pose_yaml))
    quats = [pose_cfg[i]["robot_planning_pose"][:7] for i in range(len(pose_cfg))]
    poses = [quaternion_to_mat(quat) for quat in quats] 
    reader = TopicTriggerBagReader(bag_name, *topics)
    for idx, (msgs, pose) in enumerate(zip(reader.read(), poses)):
        # if idx < 1 or idx >= 20 or idx in (7, 13):
        #     continue 
        # if idx < 61 or idx >= 80: # provide pretty solid reproj err, target 25 / 75 
        #     continue
        if idx < 41 or idx >= 60: # provide the best reproj error, target 75
            continue 
        # if idx < 21 or idx >= 40:
        #     continue 
        # if idx < 81 or idx >= 100:
        #     continue 
        # if idx < 101 or idx >= 120:
        #     continue 
        yield msg_to_img(msgs[1][1]), pose

def read_base_tt_bag(bag_name):
    topics = [
        "/tt/stopped",
        "/camera/image_color/compressed",
        "/robot_0/robot_base/end_effector_pose",
        "/track_0/position_actual",
        "/tt/angle",
    ]
    reader = TopicTriggerBagReader(bag_name, *topics)
    for msgs in reader.read():
        yield msg_to_img(msgs[1][1]), pose_msg_to_tf(msgs[2][1]), msgs[3][1].data, msgs[4][1].data 

# TODO: this is get the robot pose from pose yaml
def read_base_tt_bag_adhoc(bag_name, pose_yaml):
    pose_cfg = yaml.safe_load(open(pose_yaml))
    quats = [pose_cfg[i]["robot_planning_pose"][:7] for i in range(len(pose_cfg))]
    poses = [quaternion_to_mat(quat) for quat in quats] 
    topics = [
        "/tt/stopped",
        "/camera/image_color/compressed",
        "/track_0/position_actual",
        "/tt/control/angle_actual",
    ]
    reader = TopicTriggerBagReader(bag_name, *topics)
    for idx, (msgs, pose) in enumerate(zip(reader.read(), poses)):
        # the following are testing different batches of data
        # if idx >= 7:
        #     break
        # if idx < 7 or idx >= 14:
        #     continue 
        # if idx < 14:
        #     continue 
        yield msg_to_img(msgs[1][1]), pose, msgs[2][1].data, msgs[3][1].data

def read_joint_bag(bag_name):
    sim = build_sim_sys()
    topics = [
        "/tt/system_stopped",
        "/camera/image_raw/compressed",
        "/robot_0/robot_base/end_effector_pose",
        "/track_0/position_actual",
        "/tt/control/angle_actual",
        "/stereo/left_primary/image_raw/compressed",
        "/stereo/right_primary/image_raw/compressed",
        "/lidar_0/downsampled/velodyne_points",
        "/stereo/left_secondary/image_raw/compressed",
        "/stereo/right_secondary/image_raw/compressed",
    ]
    reader = TopicTriggerBagReader(bag_name, *topics)
    for idx, msgs in enumerate(reader.read()):
        lidar_pts = crop_lidar_roi(pc2_msg_to_array(msgs[7][1]), sim.calibration["lidar"]["tt"])
        yield msg_to_img(msgs[1][1], RGB=False), pose_msg_to_tf(msgs[2][1].pose), msgs[3][1].data, msgs[4][1].data, msg_to_img(msgs[5][1], RGB=False), \
              msg_to_img(msgs[6][1], RGB=False), lidar_pts, msg_to_img(msgs[8][1], RGB=False), msg_to_img(msgs[9][1], RGB=False)

def read_joint_bag_adhoc(bag_name, pose_yaml):
    pose_cfg = yaml.safe_load(open(pose_yaml))
    quats = [pose_cfg[i]["robot_planning_pose"][:7] for i in range(len(pose_cfg))]
    poses = [quaternion_to_mat(quat) for quat in quats] 
    tf_lidar2tt = euler_vec_to_mat([-25., -0.3, 179.4, 0, 3.44, 2.28], use_deg=True)
    topics = [
        "/tt/stopped",
        "/camera/image_color/compressed",
        "/track_0/position_actual",
        "/tt/control/angle_actual",
        "/stereo/left_primary/image_raw/compressed",
        "/stereo/right_primary/image_raw/compressed",
        "/lidar_0/cropped_points"
    ]
    reader = TopicTriggerBagReader(bag_name, *topics)
    for idx, (msgs, pose) in enumerate(zip(reader.read(), poses)):
        lidar_pts = transform_3d_pts(pc2_msg_to_array(msgs[6][1]), np.linalg.inv(tf_lidar2tt))
        yield msg_to_img(msgs[1][1]), pose, msgs[2][1].data, msgs[3][1].data, msg_to_img(msgs[4][1], RGB=False), \
              msg_to_img(msgs[5][1], RGB=False), lidar_pts 

def check_blurriness(image):
    """Compute the variance of Laplacian of the image."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(gray, cv.CV_64F).var()

def crop_lidar_roi(pts_lidar, tf_lidar2tt_init, radius=2.5, height_thres=0.05):
    pts_tt = transform_3d_pts(pts_lidar, tf_lidar2tt_init)
    z_crop = pts_tt[:, 2] > height_thres
    plane_crop = np.linalg.norm(pts_tt[:, :2], axis=1) < radius
    return pts_lidar[z_crop & plane_crop]


if __name__ == "__main__":
    # bag_path = "/home/fuhengdeng/test_data/hand_eye.bag"
    # bag_path = "/home/fuhengdeng/test_data/base_tt.bag"
    bag_path = "/home/fuhengdeng/test_data/joint_calib_2023-10-09-15-06-23.bag"
    # read_hand_eye_bag(bag_path)
    # read_intrinsic_bag(bag_path)
    # ret = read_base_tt_bag_adhoc(bag_path, "/home/fuhengdeng/data_collection_yaml/09_25/base_tt.yaml")
    # ret = read_joint_bag_adhoc(bag_path, "/home/fuhengdeng/data_collection_yaml/09_25/base_tt.yaml")
    sim = build_sim_sys()
    ret = read_joint_bag(bag_path)
    for idx, (_, _, _, _, img_left, img_right, pc) in enumerate(ret):
        pc = pc[::3]
        # numpy_to_pcd(pc, f"pcl-{idx}.pcd")
        kdtree = KDTree(pc)
        normals = []
        pts_good = []
        for pt in pc: 
            normal = plane_fitting(pt, pc, 0.4, kdtree=kdtree, fit_std_thres=0.01)
            if normal is not None:
                normals.append(normal)
                pts_good.append(pt)
        vis_points_with_normal(pts_good, normals, show_origin=False, show_point_direction=False) 