from system_calibration.IO import TopicTriggerBagReader


def read_hand_eye_bag(bag_name):
    topics = [
        "/robot_0/robot_base/end_effector_pose_stopped",
        "/camera/image_color/compressed",
        "/robot_0/robot_base/end_effector_pose"
    ]
    reader = TopicTriggerBagReader(bag_name, *topics)
    for msgs in reader.read():
        print(f"ts_diff: {abs(msgs[0][0] - msgs[1][0])/1e6} {abs(msgs[0][0] - msgs[2][0]) / 1e6}")

        
if __name__ == "__main__":
    # bag_path = "/home/fuhengdeng/test_data/hand_eye.bag"
    bag_path = "/home/fuhengdeng/fuheng.bag"
    read_hand_eye_bag(bag_path)