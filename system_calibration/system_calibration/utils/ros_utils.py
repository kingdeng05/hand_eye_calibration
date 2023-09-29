import rospy

def convert_to_unix_ns(stamp):
    return rospy.Time(stamp.secs, stamp.nsecs).to_nsec()

def convert_to_unix_ms(stamp):
    return convert_to_unix_ns(stamp) / 1e6 