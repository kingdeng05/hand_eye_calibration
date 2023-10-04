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
    # return euler_vec_to_mat([-np.pi/2, 0, -np.pi/2, 0.2, 0, 0])

def get_towercam2tt_calib(left=True):
    if left:
        return euler_vec_to_mat([-95.528, -0.331, 178.996, -0.003, 3.297, 0.405], use_deg=True)
    else:
        return euler_vec_to_mat([-96.255, 0.31, -1.769, 0.195, -3.353, 0.423], use_deg=True)

def read_cam_intrinsic(file_name=""):
    if not file_name:
        file_name = ".calibration.yaml"
    with open(file_name) as f:
        return np.array(yaml.safe_load(f)["intrinsic_vec"])

def get_img_size(tower=False):
    if tower:
        return (1920, 1200)
    else:
        return (2048, 1536)

def build_sim_sys():
    sim = SystemSimulator() 
    # add movable components
    sim.add_component("tt", Turntable(), "tt", np.eye(4), False, False)
    tf_track2tt = euler_vec_to_mat([0, 0, -180, 3.71, -0.01, 0.38], use_deg=True)
    sim.add_component("track", Track(), "tt", tf_track2tt, False, False)
    tf_robot2track = euler_vec_to_mat([-0.1, 0.6, 0, 0, 0, 0], use_deg=True)
    sim.add_component("robot", Robot(), "track", tf_robot2track, False, True)
    # add perception components 
    camera = create_camera(read_cam_intrinsic(), model="Cal3Rational") 
    sim.add_component("camera", Camera(camera, get_img_size()), "robot", read_cam2ee_calib(), False, True)
    camera = create_camera(read_cam_intrinsic(file_name=".left_primary.yaml"), model="Cal3Rational") 
    sim.add_component("lpc", Camera(camera, get_img_size(tower=True)), "tt", get_towercam2tt_calib(left=True), False, False)
    camera = create_camera(read_cam_intrinsic(file_name=".right_primary.yaml"), model="Cal3Rational") 
    sim.add_component("rpc", Camera(camera, get_img_size(tower=True)), "tt", get_towercam2tt_calib(left=False), False, False)
    # add target components
    tf_target2tt_0 = euler_vec_to_mat([-90, 0, 90, 1, 0, 0.525], use_deg=True)
    # sim.add_component("cube", ArucoCubeTarget(1.035, use_ids=(25, 50, 75, 100)), "tt", tf_target2tt_0, False, True)
    sim.add_component("cube", ArucoCubeTarget(1.035), "tt", tf_target2tt_0, False, True)
    return sim

def simulate_projection(pts_2d, text=""):
    width, height = get_img_size()
    img = np.zeros((height, width, 3))
    img_vis = draw_pts_on_img(img, pts_2d)
    img_vis = cv.resize(img_vis, (int(width/2), int(height/2)))
    if text:
        cv.putText(
            img_vis, 
            text,
            (50, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0)
        )
    cv.imshow("vis", img_vis)
    cv.waitKey(0)
    

def test_sim():
    sim = build_sim_sys()
    sim.move("track", 0)
    sim.move("robot", [0, 0, 0, 0, 0, 0.2])
    for v in np.linspace(0, 2 * np.pi, 10):
        sim.move("tt", v)
        print("cube2track", sim.get_transform("cube", "track")[:3, 3])
        simulate_projection(sim.capture("camera")[0], text="camera")
        simulate_projection(sim.capture("lpc")[0], text="left primary camera")
        simulate_projection(sim.capture("rpc")[0], text="right primary camera")

    
if __name__ == "__main__":
    test_sim()