import os
import yaml
import numpy as np 
import cv2 as cv 
from matplotlib import pyplot as plt

from collections import defaultdict

from py_kinetic_backend import Pose3, Rot3

from system_calibration.frontend import ArucoDetector
from system_calibration.simulation.components import ArucoBoardTarget 
from system_calibration.utils import calculate_reproj_error
from system_calibration.IO import read_handeye_bag
from system_calibration.backend import solve_intrinsic_rational, IntrinsicCailbrator

from read_bags import read_intrinsic_bag
from build_sim_sys import read_cam_intrinsic, build_sim_sys

np.set_printoptions(precision=3, suppress=True)

def jet_colormap(value):
    """ Converts a value in the range [0, 1] to a color using a Jet-like colormap.

    :param value: float value in the range [0, 1]
    :return: (B, G, R) color
    """
    fourValue = 4 * value
    red   = min(fourValue - 1.5, -fourValue + 4.5)
    green = min(fourValue - 0.5, -fourValue + 3.5)
    blue  = min(fourValue + 0.5, -fourValue + 2.5)

    return (
        int(255 * np.clip(blue, 0, 1)),
        int(255 * np.clip(green, 0, 1)),
        int(255 * np.clip(red, 0, 1))
    )

def draw_points_with_reprojection_error(image_size, projected_points, reprojection_errors):
    """
    Draw projected points on a white image with colors indicating their reprojection error.

    :param image_size: tuple of (width, height) specifying the size of the output image.
    :param projected_points: list of (x, y) coordinates of projected points.
    :param reprojection_errors: list of reprojection errors corresponding to each point.
    """
    # Create a white image
    img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    
    # Scale the reprojection errors to be in the range [0, 1]
    min_error = min(reprojection_errors)
    max_error = max(reprojection_errors)
    scaled_errors = [(e - min_error) / (max_error - min_error) for e in reprojection_errors]
    
    for (x, y), error in zip(projected_points, scaled_errors):
        # Generate color based on the error (you can customize the color map)
        # color = (int(255 * (1 - error)), int(255 * error), 0)
        color = jet_colormap(error) 
        
        # Draw the point
        cv.circle(img, (int(x), int(y)), 3, color, -1)
    
    # Display the image
    img = cv.resize(img, (int(image_size[0]/2), int(image_size[1]/2)))
    cv.imshow('Reprojection Errors', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def reprojection_plot(pts_2d, pts_proj):
    diff = pts_proj - pts_2d
    _, axes = plt.subplots(1, 1, squeeze=True)
    axes.scatter(diff[:, 0], diff[:, 1])
    axes.set_xlim([-20, 20])
    axes.set_ylim([-20, 20])
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    error = calculate_reproj_error(pts_2d, pts_proj)
    std = np.linalg.norm(pts_2d - pts_proj, axis=1).std()
    error_vec = np.linalg.norm(pts_2d - pts_proj, axis=1)
    print(f"50%: {np.percentile(error_vec, 50)}, 90%: {np.percentile(error_vec, 90)} 95%: {np.percentile(error_vec, 95)} max: {np.max(error_vec)}")
    axes.set_title(f"error: {error}, std: {std}")
    plt.show()

def KD_to_intrinsic_vec(K, d, model="Cal3Rational"):
    if isinstance(d, np.ndarray):
        d = d.flatten().tolist()
    if model == "Cal3Rational":
        return [K[0, 0], K[1, 1], K[0, 2], K[1, 2]] + d[:8] 
    elif model == "Cal3DS2":
        return [K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2]] + d[:4]
    elif model == "Cal3_S2":
        return [K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2]] 
    else:
        raise NotImplementedError(f"{model} not supported")

def calibrate_intrinsic(bag_path, saved_path=".calibration.yaml", debug=False):
    # load from saved path
    if not debug and os.path.exists(saved_path):
        print(f"loading intrinsic from previous result at {saved_path}")
        return np.array(yaml.safe_load(open(saved_path))["intrinsic_vec"])

    targets = {
        0: ArucoBoardTarget(3, 3, 0.166, 0.033, 0),
        9: ArucoBoardTarget(3, 3, 0.166, 0.033, 9),
        18: ArucoBoardTarget(3, 3, 0.166, 0.033, 18),
        27: ArucoBoardTarget(3, 3, 0.166, 0.033, 27),
        36: ArucoBoardTarget(3, 3, 0.166, 0.033, 36)
    }
    # targets = {
    #     0: ArucoBoardTarget(5, 5, 0.166, 0.033, 0),
    #     25: ArucoBoardTarget(5, 5, 0.166, 0.033, 25),
    #     50: ArucoBoardTarget(5, 5, 0.166, 0.033, 50),
    #     75: ArucoBoardTarget(5, 5, 0.166, 0.033, 75),
    #     100: ArucoBoardTarget(5, 5, 0.166, 0.033, 100)
    # }

    detector = ArucoDetector(vis=False)
    pts_3d_all, pts_2d_all = [], []
    views_cnt = defaultdict(int) 
    for idx, img in enumerate(read_intrinsic_bag(bag_path)):
        # 3d pts (n, 3), 2d pts (n, 1, 2)
        corners, ids = detector.detect(img)
        pts_3d = defaultdict(list) 
        pts_2d = defaultdict(list) 
        for target_id, target in targets.items():
            for corner, id in zip(corners, ids):
                ret = target.find_3d_pts_by_id(id)
                if ret is not None: 
                    pts_3d[target_id].append(ret) 
                    pts_2d[target_id].append(corner)
        for key, pts_3d_target in pts_3d.items():
            if len(pts_3d_target) >= 3:
                pts_3d_target = np.vstack(pts_3d_target, dtype=np.float32)
                pts_2d_target = np.vstack(pts_2d[key], dtype=np.float32)
                pts_2d_target = pts_2d_target.reshape(len(pts_2d_target), 1, 2)
                pts_3d_all.append(pts_3d_target)
                pts_2d_all.append(pts_2d_target)
                views_cnt[key] += 1

        # cv.imwrite(f"img{idx}.png", img)

    print(f"Using {len(pts_3d_all)} for intrinsic calibration")
    print("views distribution:\n")
    for idx, cnt in views_cnt.items():
        print(f"    target {idx}: {cnt}")

    # note that for filtering to work, it has to at least run 2 iterations 
    K, d, rvecs, tvecs, pts_3d_ft, pts_2d_ft = IntrinsicCailbrator(iteration=5, outlier_perc=99, flags=cv.CALIB_RATIONAL_MODEL).calibrate(pts_3d_all, pts_2d_all, img.shape[:2][::-1])

    if debug:
        # Compute reprojection error
        pts_2d_proj = []
        mean_error = []
        for i in range(len(pts_3d_ft)):
            imgpoints2, _ = cv.projectPoints(pts_3d_ft[i], rvecs[i], tvecs[i], K, d)
            pts_2d_proj.append(imgpoints2.reshape(-1, 2))
            error = cv.norm(pts_2d_ft[i], imgpoints2.reshape(-1, 1, 2), cv.NORM_L2) / len(imgpoints2)
            mean_error.append(error)
        pts_2d_proj = np.vstack(pts_2d_proj)
        pts_2d_ft = np.vstack(pts_2d_ft)
        pts_2d_ft = pts_2d_ft.reshape(-1, 2) 
        reprojection_plot(pts_2d_ft, pts_2d_proj)
        print(f"Total error: {np.array(mean_error).mean()}")
        print(f"min/max error: {np.array(mean_error).min()}, {np.array(mean_error).max()}")
        plt.hist(np.linalg.norm(pts_2d_proj - pts_2d_ft, axis=1), bins=50, color='blue', edgecolor='black')
        plt.title('Histogram of Reprojection Errors')
        plt.xlabel('Reprojection Error')
        plt.ylabel('Frequency')
        plt.show()
        # draw_points_with_reprojection_error(img.shape[:2][::-1], pts_2d_all, np.linalg.norm(pts_2d_proj - pts_2d_all, axis=1))

    intrinsic_vec = KD_to_intrinsic_vec(K, d, model="Cal3Rational")
    print("calibrated intrinsic:\n", intrinsic_vec)

    if not debug:
        with open(saved_path, "w+") as f:
            print(f"first time performing calibration, result saved to: {saved_path}")
            yaml.safe_dump({
                "intrinsic_vec":[float(num) for num in intrinsic_vec] 
            }, f, default_flow_style=False)
    return intrinsic_vec 

def Cal3Rational_to_KD(intr):
    return np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]]), intr[4:].reshape(-1, 8)

def Cal3DS2_to_KD(intr):
    return np.array([[intr[0], intr[2], intr[3]], [0, intr[1], intr[4]], [0, 0, 1]]), intr[5:].reshape(-1, 4)

def calibrate_intrinsic_rational(bag_path):
    targets = {
        50: ArucoBoardTarget(5, 5, 0.166, 0.033, 50),
        100: ArucoBoardTarget(5, 5, 0.166, 0.033, 100)
    }
    detector = ArucoDetector(vis=False)
    pts_3d_all, pts_2d_all = [], []
    cam_poses_all = []
    intr_vec = read_cam_intrinsic() 
    for _, (img, _, _, _, _) in enumerate(read_handeye_bag(bag_path)):
        # 3d pts (n, 3), 2d pts (n, 1, 2)
        corners, ids = detector.detect(img)
        pts_3d = defaultdict(list) 
        pts_2d = defaultdict(list) 
        for target_id, target in targets.items():
            for corner, id in zip(corners, ids):
                ret = target.find_3d_pts_by_id(id)
                if ret is not None: 
                    pts_3d[target_id].append(ret) 
                    pts_2d[target_id].append(corner)
        for key, pts_3d_target in pts_3d.items():
            if len(pts_3d_target) >= 3:
                pts_3d_target = np.vstack(pts_3d_target, dtype=np.float32)
                pts_2d_target = np.vstack(pts_2d[key], dtype=np.float32)
                pts_2d_target = pts_2d_target.reshape(len(pts_2d_target), 1, 2)
                pts_3d_all.append(pts_3d_target)
                pts_2d_all.append(pts_2d_target)
                # get the initial poses
                pts_3d_target = np.array(pts_3d_target, dtype=np.float32)
                pts_2d_target = np.array(pts_2d_target, dtype=np.float32)
                K, D = Cal3DS2_to_KD(intr_vec)
                _, rvec, tvec = cv.solvePnP(pts_3d_target, pts_2d_target, K, D)
                R = cv.Rodrigues(rvec)[0]
                cam_pose = Pose3(Rot3(R), tvec).matrix()
                cam_poses_all.append(cam_pose)

    print(f"Using {len(pts_3d_all)} for intrinsic calibration")
    intr_opt, poses_opt = solve_intrinsic_rational(intr_vec, cam_poses_all, pts_2d_all, pts_3d_all)
    print(f"optimized intrinsic: {intr_opt}")
    print(f"optimized intrinsic diff: {intr_opt - intr_vec}")

    # Compute reprojection error
    # pts_2d_proj = []
    # mean_error = []
    # for i in range(len(pts_3d_all)):
    #     imgpoints2, _ = cv.projectPoints(pts_3d_all[i], rvecs[i], tvecs[i], mtx, dist)
    #     pts_2d_proj.append(imgpoints2.reshape(-1, 2))
    #     error = cv.norm(pts_2d_all[i], imgpoints2.reshape(-1, 1, 2), cv.NORM_L2) / len(imgpoints2)
    #     mean_error.append(error)
    # pts_2d_proj = np.vstack(pts_2d_proj)
    # pts_2d_all = np.vstack(pts_2d_all)
    # pts_2d_all = pts_2d_all.reshape(-1, 2) 
    # # reprojection_plot(pts_2d_all, pts_2d_proj)
    # print(f"Total error: {np.array(mean_error).mean()}")
    # print(f"min/max error: {np.array(mean_error).min()}, {np.array(mean_error).max()}")
    # plt.hist(np.linalg.norm(pts_2d_proj - pts_2d_all, axis=1), bins=50, color='blue', edgecolor='black')
    # plt.title('Histogram of Reprojection Errors')
    # plt.xlabel('Reprojection Error')
    # plt.ylabel('Frequency')
    # plt.show()
    # # draw_points_with_reprojection_error(img.shape[:2][::-1], pts_2d_all, np.linalg.norm(pts_2d_proj - pts_2d_all, axis=1))

    # dist = dist.flatten().tolist()
    # intrinsic_vec = np.array([mtx[0, 0], mtx[1, 1], mtx[0, 1], mtx[0, 2], mtx[1, 2]] + dist[:4])
    
    return intr_opt 


if __name__ == "__main__":
    # bag_name = "/home/fuhengdeng/test_data/hand_eye.bag"
    # bag_name = "/home/fuhengdeng/test_data/hand_eye_new.bag"
    # bag_name = "/home/fuhengdeng/test_data/hand_eye_new_cube_more_angle.bag"
    bag_name = "/home/fuhengdeng/test_data/hand_eye_flatter.bag"
    # bag_name = "/home/fuhengdeng/fuheng.bag"
    calibrate_intrinsic(bag_name, debug=False)
    # calibrate_intrinsic_rational(bag_name)
    
