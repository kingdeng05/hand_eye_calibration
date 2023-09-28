import numpy as np
import yaml
import cv2 as cv 

from system_calibration.simulation import SystemSimulator
from system_calibration.simulation.components import Turntable, Track, Robot, Camera
from system_calibration.simulation.components import ArucoCubeTarget, MovableComponent
from system_calibration.simulation.components import Target
from system_calibration.utils import euler_vec_to_mat, create_camera, draw_pts_on_img

np.set_printoptions(3, suppress=True)

def read_cam2ee_calib():
    with open(".cam2ee.yaml") as f:
        return np.array(yaml.safe_load(f)["transformation"])

def read_cam_intrinsic():
    with open(".calibration.yaml") as f:
        return np.array(yaml.safe_load(f)["intrinsic_vec"])

def get_img_size():
    return (2048, 1536)

def build_sim_sys():
    sim = SystemSimulator() 
    # add movable components
    sim.add_component("tt", Turntable(), "tt", np.eye(4))
    tf_track2tt = euler_vec_to_mat([0, 0, -180, 3.71, 0, 0.38], use_deg=True)
    sim.add_component("track", Track(), "tt", tf_track2tt)
    tf_robot2track = euler_vec_to_mat([0, 0, 0, 0, 0, 0], use_deg=True)
    sim.add_component("robot", Robot(), "track", tf_robot2track)
    # add perception components 
    camera = create_camera(np.eye(4), read_cam_intrinsic(), model="Cal3Rational") 
    sim.add_component("camera", Camera(camera, get_img_size()), "robot", read_cam2ee_calib())
    # add target components
    tf_target2tt_0 = euler_vec_to_mat([-90, 0, 90, 1.8, 0, 0.525], use_deg=True)
    sim.add_component("cube", ArucoCubeTarget(1.035), "tt", tf_target2tt_0)
    return sim

def move(sim: SystemSimulator, name: str, val):
    component = sim.get_component(name)
    assert(isinstance(component, MovableComponent))
    component.move(val) 

def capture_camera(sim: SystemSimulator, name: str):
    component = sim.get_component(name)
    assert(isinstance(component, Camera))
    # get all target components
    targets = dict() 
    for n, comp in sim.components.items():
        if isinstance(comp, Target):
            targets[n] = comp
    pts_2d = []
    pts_3d = []
    for n, target in targets.items():
        tf_cam2target = sim.get_transform(name, n)
        ret = component.capture(tf_cam2target, target) 
        pts_2d.append(ret[0])
        pts_3d.append(ret[1])
    return np.vstack(pts_2d), np.vstack(pts_3d) 

def test_sim():
    sim = build_sim_sys()
    move(sim, "robot", [0, 0, 0, 0, 0, 0.3])
    move(sim, "track", 1)
    for v in np.linspace(0, np.pi, 10):
        move(sim, "tt", v)
        pts_2d, _ = capture_camera(sim, "camera")
        width, height = get_img_size()
        img = np.zeros((height, width, 3))
        img_vis = draw_pts_on_img(img, pts_2d)
        img_vis = cv.resize(img_vis, (int(width/2), int(height/2)))
        cv.imshow("vis", img_vis)
        cv.waitKey(0)

    
if __name__ == "__main__":
    test_sim()