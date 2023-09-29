from system_calibration.IO import TopicTriggerBagReader
from system_calibration.utils import convert_to_unix_ms


def read_hand_eye_bag(bag_name):
    trigger_topic = "/robot_0/robot_base/end_effector_pose_stopped"
    topics = [
        "/camera/image_color/compressed",
        "/robot_0/robot_base/end_effector_pose"
    ]
    reader = TopicTriggerBagReader(bag_name, trigger_topic, *topics)
    for msgs in reader.read():
        camera_ts = convert_to_unix_ms(msgs[0][0])
        robot_ts = convert_to_unix_ms(msgs[1][0])
        print("ts_diff: ", abs(camera_ts - robot_ts))

        
if __name__ == "__main__":
    bag_path = "/home/fuhengdeng/test_data/hand_eye.bag"
    read_hand_eye_bag(bag_path)