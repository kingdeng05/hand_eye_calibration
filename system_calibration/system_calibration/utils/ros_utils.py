import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge

from py_kinetic_backend import Pose3, Rot3

bridge = CvBridge()

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

def pose_msg_to_tf(pose_msg):
    rot = Rot3(pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z)
    return Pose3(rot, np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])).matrix()