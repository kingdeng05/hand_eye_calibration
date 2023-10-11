import numpy as np
import yaml

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

if __name__ == "__main__":
    test("static_transforms.yaml", "static_transforms_gt.yaml")