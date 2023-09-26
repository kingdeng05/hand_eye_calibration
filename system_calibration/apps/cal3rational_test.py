import numpy as np

from py_kinetic_backend import PinholeCameraCal3DS2, PinholeCameraCal3Rational
from py_kinetic_backend import Cal3DS2, Cal3Rational, Pose3

def generate_valid_pt():
    r = [
        [-2, 2],
        [-3, 3],
        [5, 50]
    ] 
    pts_all = []
    for _ in range(50):
        pt = []
        for i in range(3): 
            pt.append(np.random.uniform(*r[i]))
        pts_all.append(pt) 
    return np.array(pts_all)

def test():
    fx = fy = 1188  
    cx, cy = 1024, 768
    k1, k2, k3, k4, k5, k6 = -0.25, 0.15, 0, 0, 0, 0 
    p1, p2 = -1e-3, -1e-3
    intrinsic_cal3ds2 = [fx, fy, 0, cx, cy, k1, k2, p1, p2]
    intrinsic_cal3rational = [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2]
    pose = Pose3(np.eye(4))
    cam_ds2 = PinholeCameraCal3DS2(pose, Cal3DS2(intrinsic_cal3ds2))
    cam_rational = PinholeCameraCal3Rational(pose, Cal3Rational(intrinsic_cal3rational))
    for pt in generate_valid_pt():
        pt_ds2 = cam_ds2.project(pt)
        pt_rat = cam_rational.project(pt)
        np.testing.assert_allclose(pt_ds2, pt_rat) 

if __name__ == "__main__":
    test()
