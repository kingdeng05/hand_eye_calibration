import os
import rospy
import rosbag
import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from matplotlib import pyplot as plt

from ..frontend.aruco_detector import ArucoDetector

bridge = CvBridge()
detector = ArucoDetector(vis=False)

def blend_images(image_path1, image_path2, alpha=0.5):
    # Read the images
    image1 = cv.imread(image_path1)
    image2 = cv.imread(image_path2)

    # Check if images were successfully read
    if image1 is None or image2 is None:
        print("One of the images could not be read.")
        return

    # Make sure the images are of the same size and type
    if image1.shape != image2.shape:
        print("The images must have the same dimensions and number of channels.")
        return

    # Blend the images using the formula: new_pixel_value = alpha * pixel1 + (1 - alpha) * pixel2
    blended_image = cv.addWeighted(image1, alpha, image2, 1 - alpha, 0)

    # Display the blended image
    cv.imshow("Blended Image", blended_image)
    cv.waitKey(0)

def msg_to_pose_list(msg):
    return np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
        msg.pose.orientation.w,
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
    ])

def calc_camera_diff(corners, ids, corners_prev, ids_prev):
    diff = 0
    cnt = 0
    for id in ids:
        if id in ids_prev:
            diff += np.linalg.norm(corners_prev[ids_prev.tolist().index(id)] - corners[ids.tolist().index(id)], axis=1).mean()
            cnt += 1
    if cnt == 0:
        return 100
    else:
        return diff / cnt 

def plot_found_timestamps(bag_file_path):
    bag = rosbag.Bag(bag_file_path)
    corners_prev = None
    ids_prev = None
    pose_prev = None
    t_stop = []
    camera_t = []
    pose_t = []
    corners_diff = []
    pose_diff = []
    for topic, msg, _ in bag.read_messages():
        if "header" not in dir(msg):
            continue
        t = convert_to_unix_ns(msg.header.stamp)
        if topic == "/camera/image_color/compressed":
            corners, ids = detector.detect(msg_to_img(msg)) 
            if corners_prev is not None:
                diff = calc_camera_diff(corners, ids, corners_prev, ids_prev)
                corners_diff.append(diff)
            corners_prev = corners
            ids_prev = ids
            camera_t.append(t)
        if topic == "/robot_0/robot_base/end_effector_pose":
            pose = msg_to_pose_list(msg) 
            if pose_prev is not None:
                pose_diff.append(np.linalg.norm(pose - pose_prev))
            pose_prev = pose
            pose_t.append(t)
        if topic == "/robot_0/robot_base/end_effector_pose_stopped":
            t_stop.append(t) 

    t_camera = []
    t_pose = []
    # for t in t_stop:
    #     t_camera.append(min(camera_t, key=lambda x: np.abs(x-t) if x - 600e6 > t else np.inf))
    #     t_pose.append(min(pose_t, key=lambda x: np.abs(x-t) if x > t else np.inf))

    print("reading hand eye bag:")
    for _, _, _, t_c, t_p in read_handeye_bag(bag_file_path):
        t_camera.append(t_c)
        t_pose.append(t_p)

    _, axes = plt.subplots(2, 1, squeeze=True)
    axes[0].plot(camera_t[1:], corners_diff, picker=True)
    for val in t_stop:
        axes[0].axvline(x=val, color='r')
    for val in t_camera:
        axes[0].axvline(x=val, color='g')
    axes[0].set_title("camera")
    axes[1].plot(pose_t[1:], pose_diff, picker=True)
    for val in t_stop:
        axes[1].axvline(x=val, color='r')
    for val in t_pose:
        axes[1].axvline(x=val, color='g')
    axes[1].set_title("pose")
    plt.show()

def convert_to_unix_ns(stamp):
    return rospy.Time(stamp.secs, stamp.nsecs).to_nsec()

def convert_to_unix_ms(stamp):
    return convert_to_unix_ns(stamp) / 1e6 

def msg_to_img(img_msg, compressed=True):
    if compressed:
        # compressed image parsing
        img = cv.imdecode(np.frombuffer(img_msg.data, np.uint8), cv.IMREAD_COLOR)
    else:
        # raw msg parsing
        img = np.array(bridge.imgmsg_to_cv2(img_msg, "bgr8"))
    return  img

def check_camera_msg(corners_1, ids_1, corners_2, ids_2, pixel_tol=2):
    for id_1 in ids_1:
        if id_1 in ids_2:
            c_1 = corners_1[ids_1.tolist().index(id_1)]
            c_2 = corners_2[ids_2.tolist().index(id_1)]
            for i in range(4):
                if np.linalg.norm(c_1[i] - c_2[i]) > pixel_tol: 
                    print(np.linalg.norm(c_1[i] - c_2[i]))
                    return False
    return True

def compare_pose_diff(pose_msg_1, pose_msg_2, pos_tol=1e-4, orientation_tol=1e-4):
    return abs(pose_msg_1.position.x - pose_msg_2.position.x) < pos_tol and \
           abs(pose_msg_1.position.y - pose_msg_2.position.y) < pos_tol and \
           abs(pose_msg_1.position.z - pose_msg_2.position.z) < pos_tol and \
           abs(pose_msg_1.orientation.w - pose_msg_2.orientation.w) < orientation_tol and \
           abs(pose_msg_1.orientation.x - pose_msg_2.orientation.x) < orientation_tol and \
           abs(pose_msg_1.orientation.y - pose_msg_2.orientation.y) < orientation_tol and \
           abs(pose_msg_1.orientation.z - pose_msg_2.orientation.z) < orientation_tol

def check_pose_msg_stable(ros_msgs):
    ros_msg_baseline = ros_msgs[0]
    for idx, ros_msg in enumerate(ros_msgs):
        if not compare_pose_diff(ros_msg.pose, ros_msg_baseline.pose):
            print(idx, ":\n", ros_msg.pose, "\n", ros_msg_baseline.pose)
            return False
    return True

def find_when_robot_stop(msgs):
    for idx, ros_msg in enumerate(msgs):
        if idx == 0:
            continue
        if compare_pose_diff(ros_msg.pose, msgs[idx-1].pose):
            return idx
    else:
        return -1

def dump_images(bag_file_path, folder="."):
    bag = rosbag.Bag(bag_file_path)
    cnt = 0
    for topic, msg, t in bag.read_messages():
        if topic == "/camera/image_color":
            cv.imwrite(os.path.join(folder, "{}.png".format(cnt)), msg_to_img(msg))
            cnt += 1

def analyze_time(bag_file_path):
    bag = rosbag.Bag(bag_file_path)
    stop_t = []
    camera_t = []
    ee_t = []
    img_topics = [] 
    corners = []
    ids = []
    ee_topics = [] 
    for topic, msg, t in bag.read_messages():
        t = t.to_nsec()
        if topic == "/robot_0/robot_base/end_effector_pose_stopped": 
            stop_t.append(t)
        if topic == "/camera/image_color":
            camera_t.append(t) 
            c, i = detector.detect(msg_to_img(msg))
            corners.append(c)
            ids.append(i)
        if topic == "/robot_0/robot_base/end_effector_pose":
            ee_t.append(t)
            ee_topics.append(msg)

    for t in stop_t:
        t_cam = min(camera_t, key=lambda x: np.abs(x-t) if x - 500e6 > t else np.inf)
        t_ee = min(ee_t, key=lambda x: np.abs(x-t) if x - 500e6 > t else np.inf)
        t_cam_index = camera_t.index(t_cam)
        t_ee_index = ee_t.index(t_ee)
        check_camera_flag =  check_camera_msg(corners[t_cam_index], ids[t_cam_index], corners[t_cam_index+1], ids[t_cam_index+1])
        print(check_camera_flag) 
        if not check_camera_flag:
            blend_images(f"{t_cam_index}.png", f"{t_cam_index+1}.png") 
        # print(check_pose_msg_stable(ee_topics[t_ee_index:t_ee_index+3])) 
        print((t_cam - t_ee) / 1e6, (t_cam - t) / 1e6)
        # print(find_when_robot_stop(ee_topics[t_ee_index:t_ee_index+20]))
    
def analyze_robot_pose(bag_file_path):
    bag = rosbag.Bag(bag_file_path)
    stop_t = []
    camera_t = []
    ee_t = []
    for topic, _, t in bag.read_messages():
        t = t.to_nsec()
        if topic == "/robot_0/robot_base/end_effector_pose_stopped": 
            stop_t.append(t)
        if topic == "/camera/image_color":
            camera_t.append(t) 
        if topic == "/robot_0/robot_base/end_effector_pose":
            ee_t.append(t)

    for t in stop_t:
        t_cam = min(camera_t, key=lambda x: np.abs(x-t) if x > t else np.inf)
        t_ee = min(ee_t, key=lambda x: np.abs(x-t) if x > t else np.inf)
        print(np.abs(t_cam - t_ee) / 1e6, np.abs(t_cam - t) / 1e6)

def read_handeye_bag(bag_file_path):
    end_effector_stop_time = None
    next_image_after_stop = None
    next_pose_after_stop = None
    corners_prev = None 
    ids_prev = None 
    pose_prev = None

    bag = rosbag.Bag(bag_file_path)
    diff_camera_robot = []
    diff_camera_end = []
    for topic, msg, t in bag.read_messages():
        if topic == '/robot_0/robot_base/end_effector_pose_stopped':
            end_effector_stop_time = convert_to_unix_ns(msg.header.stamp) 

        if end_effector_stop_time is not None: 
            if topic == '/camera/image_color/compressed' and convert_to_unix_ns(msg.header.stamp) > end_effector_stop_time:
                corners, ids = detector.detect(msg_to_img(msg))
                if corners_prev is not None:
                    diff = calc_camera_diff(corners, ids, corners_prev, ids_prev)
                    if diff < 1 and next_image_after_stop is None:
                        next_image_after_stop = msg 
                corners_prev = corners
                ids_prev = ids
            if topic == '/robot_0/robot_base/end_effector_pose' and convert_to_unix_ns(msg.header.stamp) > end_effector_stop_time:
                if next_pose_after_stop is None:
                    if pose_prev is not None and compare_pose_diff(pose_prev, msg.pose):
                            next_pose_after_stop = msg
                    pose_prev = msg.pose
                    next_pose_after_stop = msg

            if next_image_after_stop is not None and next_pose_after_stop is not None:
                image = msg_to_img(next_image_after_stop, compressed=True) 
                pose = next_pose_after_stop.pose  
                cam_ts = convert_to_unix_ns(next_image_after_stop.header.stamp)
                pose_ts = convert_to_unix_ns(next_pose_after_stop.header.stamp)
                diff_camera_robot.append(np.abs(cam_ts - pose_ts))
                diff_camera_end.append(np.abs(cam_ts - end_effector_stop_time / 1e6))
                yield image, pose, end_effector_stop_time, cam_ts, pose_ts 
                end_effector_stop_time = None
                next_image_after_stop = None
                next_pose_after_stop = None
                corners_prev = None
                ids_prev = None

    # plt.figure()
    # plt.plot(diff_camera_robot)
    # plt.plot(diff_camera_end)
    # plt.legend(["diff_camera_robot", "diff_camera_end"])
    # plt.show()

    bag.close()

if __name__ == "__main__":
    # bag_file_path = '/home/fuhengdeng/fuheng.bag'
    bag_file_path = '/home/fuhengdeng/test_data/hand_eye.bag'

    # it = read_handeye_bag(bag_file_path)

    # for idx, (img, pose) in enumerate(it):
    #     print(f"**************{idx}*************")
    #     print(img.shape) 
    #     print(f"pose: {pose.position}, {pose.orientation}") 

    # analyze_time(bag_file_path)
    # dump_images(bag_file_path)
    plot_found_timestamps(bag_file_path)

