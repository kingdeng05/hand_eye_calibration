import numpy as np
import yaml

from system_calibration.utils import euler_vec_to_mat, mat_to_euler_vec

np.set_printoptions(precision=3, suppress=True)

ANGLE_THRES  = 0.5 
TRANS_THRES  = 0.1 

def angle_diff(angle1, angle2):
    diff = (angle1 - angle2 + 180) % 360 - 180
    return diff

def test(file_path, gt_file_path):
    with open(file_path) as f:
        cfg = yaml.safe_load(f)
    with open(gt_file_path) as f:
        cfg_gt = yaml.safe_load(f)
    for k, _ in cfg_gt.items():
        if "_to_" in k:
            if any([s in k for s in ["roll", "pitch", "yaw"]]):
                diff = angle_diff(cfg_gt[k], cfg[k])
                if diff > ANGLE_THRES:
                    print(f"{k} value {diff} larger than threshold: {ANGLE_THRES}")
            if any([s in k for s in ["x_T", "y_T", "z_T"]]):
                diff = np.abs(cfg_gt[k] - cfg[k])
                if diff > TRANS_THRES:
                    print(f"{k} value {diff} larger than threshold: {TRANS_THRES}")

def get_transform(cfg, name):
    return euler_vec_to_mat([
            cfg.get(name + "_roll", 0),
            cfg.get(name + "_pitch", 0),
            cfg.get(name + "_yaw", 0),
            cfg.get(name + "_x_T", 0),
            cfg.get(name + "_y_T", 0),
            cfg.get(name + "_z_T", 0),
    ], use_deg=True)

def get_base2tt_transform_diff(file_path, gt_file_path):
    def get_base2tt_mat(cfg):
        base2track_mat = get_transform(cfg, "/kinetic_parameters/track_frame_to_track_base") 
        track2tt_mat = get_transform(cfg, "/kinetic_parameters/tt_0_to_track") 
        return track2tt_mat @ base2track_mat
    with open(file_path) as f:
        cfg = yaml.safe_load(f)
    with open(gt_file_path) as f:
        cfg_gt = yaml.safe_load(f)
    base2tt = get_base2tt_mat(cfg)
    base2tt_gt = get_base2tt_mat(cfg_gt)
    print(mat_to_euler_vec(np.linalg.inv(base2tt_gt) @ base2tt))

def get_stereo_extrinsic_diff(file_path, gt_file_path):
    def get_stereo_extrinsic(cfg, left=True):
        tower = "left" if left else "right"
        primary = get_transform(cfg, f"/kinetic_parameters/tt_0_to_camera_{tower}_primary") 
        secondary = get_transform(cfg, f"/kinetic_parameters/tt_0_to_camera_{tower}_secondary") 
        return np.linalg.inv(primary) @ secondary 
    with open(file_path) as f:
        cfg = yaml.safe_load(f)
    with open(gt_file_path) as f:
        cfg_gt = yaml.safe_load(f)
    ext_left = get_stereo_extrinsic(cfg, left=True) 
    ext_left_gt = get_stereo_extrinsic(cfg_gt, left=True) 
    print("left extrinsic diff: ", mat_to_euler_vec(np.linalg.inv(ext_left_gt) @ ext_left))
    ext_right = get_stereo_extrinsic(cfg, left=False) 
    ext_right_gt = get_stereo_extrinsic(cfg_gt, left=False) 
    print("right extrinsic diff: ", mat_to_euler_vec(np.linalg.inv(ext_right_gt) @ ext_right))


if __name__ == "__main__":
    # test("static_transforms.yaml", "static_transforms_gt.yaml")
    get_stereo_extrinsic_diff("static_transforms.yaml", "static_transforms_gt.yaml")
    # get_base2tt_transform_diff("static_transforms.yaml", "static_transforms_gt.yaml")

