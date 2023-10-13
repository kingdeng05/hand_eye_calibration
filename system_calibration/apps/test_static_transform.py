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

def get_base2tt_transform_diff(file_path, gt_file_path):
    def get_base2tt_mat(cfg):
        base2track_mat = euler_vec_to_mat([
            cfg["/kinetic_parameters/track_frame_to_track_base_roll"],
            cfg["/kinetic_parameters/track_frame_to_track_base_pitch"],
            cfg["/kinetic_parameters/track_frame_to_track_base_yaw"],
            0,
            0,
            0
        ], use_deg=True)
        track2tt_mat = euler_vec_to_mat([
            cfg["/kinetic_parameters/tt_0_to_track_roll"],
            cfg["/kinetic_parameters/tt_0_to_track_pitch"],
            cfg["/kinetic_parameters/tt_0_to_track_yaw"],
            cfg["/kinetic_parameters/tt_0_to_track_x_T"],
            cfg["/kinetic_parameters/tt_0_to_track_y_T"],
            cfg["/kinetic_parameters/tt_0_to_track_z_T"]
        ], use_deg=True)
        return track2tt_mat @ base2track_mat

    with open(file_path) as f:
        cfg = yaml.safe_load(f)
    with open(gt_file_path) as f:
        cfg_gt = yaml.safe_load(f)
    base2tt = get_base2tt_mat(cfg)
    base2tt_gt = get_base2tt_mat(cfg_gt)
    print(mat_to_euler_vec(np.linalg.inv(base2tt_gt) @ base2tt))


if __name__ == "__main__":
    # test("static_transforms.yaml", "static_transforms_gt.yaml")
    get_base2tt_transform_diff("static_transforms.yaml", "static_transforms_gt.yaml")

