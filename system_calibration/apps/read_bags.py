import yaml
import cv2 as cv
from matplotlib import pyplot as plt

from system_calibration.IO import TopicTriggerBagReader
from system_calibration.frontend import ArucoDetector 
from system_calibration.utils import msg_to_img, pose_msg_to_tf, quaternion_to_mat



def read_intrinsic_bag(bag_name):
    topics = [
        "/robot_0/robot_base/end_effector_pose_stopped",
        "/camera/image_color/compressed",
    ]
    reader = TopicTriggerBagReader(bag_name, *topics)
    for msgs in reader.read():
        img = msg_to_img(msgs[1][1])
        yield img

def read_hand_eye_bag(bag_name):
    topics = [
        "/robot_0/robot_base/end_effector_pose_stopped",
        "/camera/image_color/compressed",
        "/robot_0/robot_base/end_effector_pose",
    ]
    reader = TopicTriggerBagReader(bag_name, *topics)
    for msgs in reader.read():
        yield msg_to_img(msgs[1][1]), pose_msg_to_tf(msgs[2][1]) 

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

def check_blurriness(image):
    """Compute the variance of Laplacian of the image."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(gray, cv.CV_64F).var()

def plot_blurriness(bag_name):
    topics = ["/camera/image_color/compressed"]
    reader = TopicTriggerBagReader(bag_name, *topics)
    vals = []
    for msgs in reader.read():
        vals.append(check_blurriness(msg_to_img(msgs[0][1])))
    plt.plot(vals)
    plt.show()

     
if __name__ == "__main__":
    # bag_path = "/home/fuhengdeng/test_data/hand_eye.bag"
    bag_path = "/home/fuhengdeng/test_data/base_tt.bag"
    # read_hand_eye_bag(bag_path)
    # read_intrinsic_bag(bag_path)
    ret = read_base_tt_bag_adhoc(bag_path, "/home/fuhengdeng/data_collection_yaml/09_25/base_tt.yaml")
    for _ in ret:
        continue