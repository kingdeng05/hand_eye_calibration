import os
import json
import numpy as np
import yaml
import cv2 as cv 

from system_calibration.simulation import SystemSimulator
from system_calibration.simulation.components import Turntable, Track, Robot, Camera, LiDAR
from system_calibration.simulation.components import ArucoCubeTarget, ArucoCubeTargetV2
from system_calibration.utils import euler_vec_to_mat, create_camera, draw_pts_on_img

DATA_PATH = "./data"

np.set_printoptions(3, suppress=True)

def get_secondary_to_primary(file_path):
    with open(file_path) as f:
        cfg = json.load(f)
        tf_second_to_primary = np.array(cfg["calibration"][1]["pose"]["data"]).reshape(4, 4)
    return tf_second_to_primary

def read_cam2ee_calib(file_path=os.path.join(DATA_PATH, "cam2ee.yaml")):
    if os.path.exists(file_path):
        with open(file_path) as f:
            return np.array(yaml.safe_load(f)["transformation"])
    else:
        return euler_vec_to_mat([-np.pi/2, 0, -np.pi/2, 0.2, 0, 0]) 

def get_towercam2tt_calib(left=True):
    if left:
        return euler_vec_to_mat([-95.528, -0.331, 178.996, -0.003, 3.297, 0.405], use_deg=True)
    else:
        return euler_vec_to_mat([-96.255, 0.31, -1.769, 0.195, -3.353, 0.423], use_deg=True)

def read_cam_intrinsic(file_name=os.path.join(DATA_PATH, "calibration.yaml")):
    with open(file_name) as f:
        return np.array(yaml.safe_load(f)["intrinsic_vec"])

def read_tower_cam_intrinsic(file_path, primary=True):
    with open(file_path) as f:
        cfg = json.load(f)
        idx = 0 if primary else 1
        cam_mat = np.array(cfg["calibration"][idx]["intrinsics"]["camera_matrix"]["data"])
        intrinsic_vec = cam_mat[[0, 4, 2, 5]].tolist() 
        distortion_vec = cfg["calibration"][idx]["intrinsics"]["distortion_coefficients"]["data"] 
    return intrinsic_vec + distortion_vec

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
    sim.add_component("camera", Camera(read_cam_intrinsic(), get_img_size()), "robot", read_cam2ee_calib(), False, True)
    lpc_intrinsic = read_tower_cam_intrinsic(os.path.join(DATA_PATH, "stereo_left.json"))
    sim.add_component("lpc", Camera(lpc_intrinsic, get_img_size(tower=True)), "tt", get_towercam2tt_calib(left=True), False, False)
    rpc_intrinsic = read_tower_cam_intrinsic(os.path.join(DATA_PATH, "stereo_right.json"))
    sim.add_component("rpc", Camera(rpc_intrinsic, get_img_size(tower=True)), "tt", get_towercam2tt_calib(left=False), False, False)
    lsc_intrinsic = read_tower_cam_intrinsic(os.path.join(DATA_PATH, "stereo_left.json"), primary=False)
    tf_lsc2lpc = get_secondary_to_primary(os.path.join(DATA_PATH, "stereo_left.json"))
    sim.add_component("lsc", Camera(lsc_intrinsic, get_img_size(tower=True)), "lpc", tf_lsc2lpc, False, False)
    rsc_intrinsic = read_tower_cam_intrinsic(os.path.join(DATA_PATH, "stereo_right.json"), primary=False)
    tf_rsc2rpc = get_secondary_to_primary(os.path.join(DATA_PATH, "stereo_right.json"))
    sim.add_component("rsc", Camera(rsc_intrinsic, get_img_size(tower=True)), "rpc", tf_rsc2rpc, False, False)
    sim.add_component("lidar", LiDAR(), "tt", euler_vec_to_mat([-25., -0.3, 179.4, 0, 3.44, 2.28], use_deg=True), False, False)
    # add target components
    # tf_target2tt_0 = euler_vec_to_mat([-90, 0, 90, 1, 0, 0.525], use_deg=True)
    # sim.add_component("cube", ArucoCubeTarget(1.035), "tt", tf_target2tt_0, False, True)
    tf_target2tt_0 = euler_vec_to_mat([-90, 0, 90, 1, 0, 0.333738], use_deg=True)
    sim.add_component("cube", ArucoCubeTargetV2(0.6561), "tt", tf_target2tt_0, False, True)
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
        simulate_projection(sim.capture("camera")[0], text="camera")
        simulate_projection(sim.capture("lpc")[0], text="left primary camera")
        simulate_projection(sim.capture("rpc")[0], text="right primary camera")
        simulate_projection(sim.capture("lsc")[0], text="left secondary camera")
        simulate_projection(sim.capture("rsc")[0], text="right secondary camera")

    
if __name__ == "__main__":
    test_sim()