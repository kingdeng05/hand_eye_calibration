import os
import yaml
import numpy as np
import cv2 as cv
from copy import deepcopy 
from scipy.spatial import KDTree
from matplotlib import pyplot as plt

from py_kinetic_backend import Surfel3, Unit3
from system_calibration.simulation import *
from system_calibration.frontend import ArucoDetector
from system_calibration.backend import solve_joint_calib, solve_joint_calib_2 
from system_calibration.simulation.components.perception_components import is_plane_visible
from system_calibration.utils import euler_vec_to_mat, mat_to_euler_vec, tf_mat_diff
from system_calibration.utils import transfer_3d_pts_to_img, calculate_reproj_error, wrap_around_rad
from system_calibration.utils import visualize_reprojection, draw_pts_on_img, plane_fitting 
from system_calibration.utils import VideoRecorder, transform_3d_pts 

from build_sim_sys import build_sim_sys, simulate_projection
from build_sim_sys import read_cam_intrinsic, read_cam2ee_calib, get_img_size 
from read_bags import read_joint_bag_adhoc

VIS = False 

np.random.seed(5)
np.set_printoptions(precision=3, suppress=True)

def print_reproj_stats(pts_2d, pts_proj):
    local_re = []
    for pts_2d_frame, pts_proj_frame in zip(pts_2d, pts_proj):
        err_frame = calculate_reproj_error(pts_2d_frame, pts_proj_frame)
        local_re.append(err_frame)
    global_re = np.linalg.norm(np.vstack(pts_2d) - np.vstack(pts_proj), axis=1)
    print(f"local reproj error:\n  mean: {np.mean(local_re)}, min: {np.min(local_re)}, max: {np.max(local_re)}, std: {np.std(local_re)}")
    print(f"global reproj error:\n  mean: {np.mean(global_re)}, min: {np.min(global_re)}, max: {np.max(global_re)}, std: {np.std(global_re)}")

def print_vec(mat):
    print(mat_to_euler_vec(mat, use_deg=True))

def get_gt():
    sim = build_sim_sys()
    gt = dict()
    gt["track2tt"] = sim.calibration["track"]["tt"] 
    gt["base2track"] = sim.calibration["robot"]["track"] 
    gt["intrinsic"] = read_cam_intrinsic()
    gt["intrinsic_lpc"] = read_cam_intrinsic(file_name=".left_primary.yaml")
    gt["intrinsic_rpc"] = read_cam_intrinsic(file_name=".right_primary.yaml")
    gt["cam2ee"] = read_cam2ee_calib() 
    gt["target2tt_0"] = sim.calibration["cube"]["tt"] 
    gt["lpc2tt"] = sim.calibration["lpc"]["tt"] 
    gt["rpc2tt"] = sim.calibration["rpc"]["tt"] 
    gt["lidar2tt"] = sim.calibration["lidar"]["tt"] 
    return gt

def make_pts_pair(pts_2d, pts_3d):
    return {
        "2d": pts_2d,
        "3d": pts_3d
    }

def get_2d_3d_matches(corners_2d, ids_2d, targets):
    pts_3d, pts_2d = [], []
    for corner,  id in zip(corners_2d, ids_2d):
        pts_target = targets.find_3d_pts_by_id(id) 
        if pts_target is not None:
            pts_3d.append(pts_target)
            pts_2d.append(corner)
    pts_3d = np.vstack(pts_3d)
    pts_2d = np.vstack(pts_2d)
    return make_pts_pair(pts_2d, pts_3d)

def plot_clusters_and_connections(cluster1, cluster2):
    # Ensure clusters are of same size
    assert cluster1.shape == cluster2.shape, "Clusters should have the same number of points"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the two clusters of points
    ax.scatter(cluster1[:,0], cluster1[:,1], cluster1[:,2], c='r', marker='o')
    ax.scatter(cluster2[:,0], cluster2[:,1], cluster2[:,2], c='b', marker='^')

    # Draw lines connecting the points between the two clusters
    for point1, point2 in zip(cluster1, cluster2):
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], 'k-')
    plt.show()

def get_3d_surfel_matches(pts_3d, target, tf_lidar2target, downsample_rate=5, search_radius=0.4, pt_asso_thres=1):
    pts_3d_downsampled = pts_3d[::downsample_rate]
    target_planes = dict() 
    for target_id in target.use_ids:
        tf_target2plane = euler_vec_to_mat(target.frames[target_id], use_deg=False)
        tf_lidar2plane = tf_target2plane @ tf_lidar2target
        plane_in_lidar = (np.linalg.inv(tf_lidar2plane)[:3, :3].dot(np.array([0, 0, 1]).reshape(-1, 1))).flatten()
        pts_target_center_lidar = transform_3d_pts([target.get_pts_3d_by_id(target_id).mean(axis=0)], np.linalg.inv(tf_lidar2target))[0]
        plane_surfel = Surfel3(pts_target_center_lidar, Unit3(plane_in_lidar), 1.)
        target_planes[target_id] = plane_surfel
    pts = []
    surfels = []
    kdtree = KDTree(pts_3d_downsampled)
    for pt_lidar in pts_3d_downsampled:
        normal = plane_fitting(pt_lidar, pts_3d_downsampled, search_radius, kdtree=kdtree, fit_std_thres=0.01)
        best_surfel_id = None
        best_normal_similarity = -np.inf
        for target_id, plane_surfel in target_planes.items():
            target_plane_normal = plane_surfel.normal.point3()
            if normal is not None and \
               np.linalg.norm(pt_lidar - plane_surfel.center) < pt_asso_thres:
               #is_plane_visible(target_plane_normal, pt_lidar):
                # make it absolute value because the normal direction could be 180 flipped
                normal_similarity = np.abs(normal.dot(target_plane_normal))
                if normal_similarity > best_normal_similarity:
                    best_surfel_id = target_id
                    best_normal_similarity = normal_similarity
        if best_surfel_id is not None:
            pts.append(pt_lidar)
            # remember to record the surfel under target base frame,  not plane nor lidar.
            tf_target2plane = euler_vec_to_mat(target.frames[best_surfel_id], use_deg=False)
            plane_in_target = (np.linalg.inv(tf_target2plane)[:3, :3].dot(np.array([0, 0, 1]).reshape(-1, 1))).flatten()
            surfels.append(Surfel3(
                target.get_pts_3d_by_id(best_surfel_id).mean(axis=0),
                Unit3(plane_in_target),
                1.
            ))
    pts = np.array(pts)
    # centers = np.array([surfel.center for surfel in surfels])
    # plot_clusters_and_connections(pts, centers)
    return {
        "pts": pts,
        "surfels": surfels 
    }

def sim_bag_read():
    sim = build_sim_sys()
    width, height = get_img_size()
    width_tower, height_tower = get_img_size(tower=True)
    ee_pose_vec = [0, 0, 0, 0.6, 0, 0.8]
    sim.move("robot", ee_pose_vec) 
    for tr in [0.5, 1, 2]:
        sim.move("track", tr)
        for tt in np.linspace(0, 2 * np.pi, 7):
            sim.move("tt", tt)
            pts_2d, pts_3d = sim.capture("camera")
            pts_2d_lpc, pts_3d_lpc = sim.capture("lpc")
            pts_2d_rpc, pts_3d_rpc = sim.capture("rpc")
            if VIS:
                simulate_projection(pts_2d)
                simulate_projection(pts_2d_lpc)
                simulate_projection(pts_2d_rpc)
            yield np.zeros((height, width, 3)), \
                  np.zeros((height_tower, width_tower, 3)), \
                  np.zeros((height_tower, width_tower, 3)), \
                  make_pts_pair(pts_2d, pts_3d), \
                  make_pts_pair(pts_2d_lpc, pts_3d_lpc), \
                  make_pts_pair(pts_2d_rpc, pts_3d_rpc), \
                  euler_vec_to_mat(ee_pose_vec), \
                  track_reading_to_transform(tr), \
                  tt_reading_to_transform(tt) 

def read_bag(bag_path):
    detector = ArucoDetector(vis=VIS)
    targets = ArucoCubeTarget(1.035)
    tf_lidar2tt = np.linalg.inv(euler_vec_to_mat([-25., -0.3, 179.4, 0, 3.44, 2.28], use_deg=True))
    sim = build_sim_sys()
    tf_tt2target = sim.get_transform("tt", "cube")
    # TODO: adhoc to provide pose files
    for msgs in read_joint_bag_adhoc(bag_path, "/home/fuhengdeng/data_collection_yaml/09_25/base_tt.yaml"):
        corners, ids = detector.detect(msgs[0]) 
        corners_lpc, ids_lpc = detector.detect(msgs[4]) 
        corners_rpc, ids_rpc = detector.detect(msgs[5]) 
        if len(ids) == 0 and len(ids_lpc) == 0 and len(ids_rpc) == 0:
            continue
        tt_tf = tt_reading_to_transform(msgs[3]) 
        tf_lidar2target = tf_tt2target @ np.linalg.inv(tt_tf) @ tf_lidar2tt
        yield msgs[0], \
              cv.cvtColor(msgs[4], cv.COLOR_GRAY2RGB), \
              cv.cvtColor(msgs[5], cv.COLOR_GRAY2RGB), \
              msgs[6], \
              get_2d_3d_matches(corners, ids, targets), \
              get_2d_3d_matches(corners_lpc, ids_lpc, targets), \
              get_2d_3d_matches(corners_rpc, ids_rpc, targets), \
              get_3d_surfel_matches(msgs[6], targets, tf_lidar2target), \
              msgs[1], \
              track_reading_to_transform(msgs[2]), \
              tt_tf

def perturb_pose3(pose_mat, var, current=True):
    assert(len(var) == 6)
    pert_vec = [np.random.normal(0, v) for v in var]
    pert_mat = euler_vec_to_mat(pert_vec)
    if current:
        return pose_mat @ pert_mat  
    else:
        return pert_mat @ pose_mat

def perturb_pts(pts, var):
    # pts must be in (n, 2) or (n, 3)
    assert pts.shape[1] == len(var)
    noises = []
    for v in var:
        noises.append(np.random.normal(0, v, len(pts)))
    noises = np.array(noises).T
    return pts + noises 

def track_reading_to_transform(track_reading):
    # track moves in x direction, 0 is at the very begining
    # so only negative values
    tf = np.eye(4)
    # take negative sign here because x is pointing forward but track moves backward
    tf[0, 3] = -track_reading 
    return tf

def tt_reading_to_transform(tt_reading):
    tt_reading = wrap_around_rad(tt_reading) 
    vec = [0, 0, tt_reading, 0, 0, 0]
    return euler_vec_to_mat(vec) 

def append_dict_list(d, key, val):
    if key not in d:
        d[key] = [] 
    d[key].append(val)

def get_data(bag_path):
    # read data in either sim mode or actual bag
    if bag_path is not None and os.path.exists(bag_path):
        print(f"Loading bag from {bag_path}")
        read = read_bag(bag_path)
    else:
        print(f"No bag path is given or {bag_path} doesn't exist, running simulation...")
        read = sim_bag_read()
    return read
    
def calibrate_joint_calib(bag_path=None, perturb=False, debug=False):
    track_tfs = []
    tt_tfs = []
    pts_rc_all, pts_lpc_all, pts_rpc_all = [], [], []
    initials = get_gt()

    rc_recorder = VideoRecorder(2, filename="rc.mp4")
    lpc_recorder = VideoRecorder(2, filename="lpc.mp4")
    rpc_recorder = VideoRecorder(2, filename="rpc.mp4")

    for _, (_, _, _, pts_rc, pts_lpc, pts_rpc, hand_pose, track_tf, tt_tf, _) in enumerate(get_data(bag_path)):
        track_tfs.append(track_tf)
        tt_tfs.append(tt_tf)
        pts_rc_all.append(pts_rc)
        pts_lpc_all.append(pts_lpc)
        pts_rpc_all.append(pts_rpc)
    initials["ee2base"] = hand_pose # note that the hand is not moving during calibration

    # perturbation test
    if perturb:
        pert = deepcopy(initials)
        pert["track2tt"] = perturb_pose3(pert["track2tt"], [1e-3, 1e-3, 0, 0.1, 0.1, 1e-2]) 
        pert["base2track"] = perturb_pose3(pert["base2track"], [1e-5, 1e-5, 0.1, 0, 0, 0]) 
        track_tfs = [perturb_pose3(tr_tf, [0, 0, 0, 1e-2, 0, 0]) for tr_tf in track_tfs]
        tt_tfs = [perturb_pose3(tt_tf, [0, 0, 2e-3, 0, 0, 0]) for tt_tf in tt_tfs]
        pert["ee2base"] = perturb_pose3(initials["ee2base"], [1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4])
        pts_rc_all = [{"2d": perturb_pts(pts["2d"], [1, 1]), "3d": pts["3d"]} for pts in pts_rc_all] 
        pts_lpc_all = [{"2d": perturb_pts(pts["2d"], [1, 1]), "3d": pts["3d"]} for pts in pts_lpc_all] 
        pts_rpc_all = [{"2d": perturb_pts(pts["2d"], [1, 1]), "3d": pts["3d"]} for pts in pts_rpc_all] 
        pert["target2tt"] = perturb_pose3(pert["track2tt"], [1e-2, 1e-5, 1e-5, 1e-1, 1e-4, 1e-1])
    else:
        pert = initials

    print("frontend features collection finished, running optimizatin...")
    ret = solve_joint_calib(pts_rc_all, pts_lpc_all, pts_rpc_all, track_tfs, tt_tfs, pert)
    for key, val in ret.items():
        if "2" in key: # only the static transforms
            print(f"diff of {key}: ", mat_to_euler_vec(np.linalg.inv(initials[key]) @ val))
            # print(f"diff of {key}: ", mat_to_euler_vec(initials[key] @ np.linalg.inv(val)))
    print("track2tt: ", mat_to_euler_vec(ret["track2tt"]))
    print("base2track: ", mat_to_euler_vec(ret["base2track"]))

    # evaluate the result
    pts_2d_rc, pts_proj_rc = [], [] 
    pts_2d_lpc, pts_proj_lpc = [], []
    pts_2d_rpc, pts_proj_rpc = [], []
    for idx, ((img_rc, img_lpc, img_rpc, pts_rc, pts_lpc, pts_rpc, ee2base_meas, track_tf_meas, tt_tf_meas, _), track_tf, tt_tf) in enumerate(zip(get_data(bag_path), ret["track_tfs"], ret["tt_tfs"])):
        tf_target2tt_i = tt_tf @ ret["target2tt_0"] 

        # project on the robot camera
        tf_cam2tt = ret["track2tt"] @ track_tf @ ret["base2track"] @ ret["ee2base"] @ initials["cam2ee"]
        tf_cam2target = np.linalg.inv(tf_target2tt_i) @ tf_cam2tt
        tf_cam2tt_init = initials["track2tt"] @ track_tf @ initials["base2track"] @ ret["ee2base"] @ initials["cam2ee"] 
        pts_2d, pts_proj, img_rc = evaluate_reproj(img_rc, pts_rc["3d"], pts_rc["2d"], tf_cam2target, tf_cam2tt, tf_cam2tt_init, \
                                                   initials["intrinsic"], tt_tf, track_tf, tt_tf_meas, track_tf_meas, debug)
        rc_recorder.add_frame(img_rc, text=f"frame-{idx}")
        pts_2d_rc.append(pts_2d)
        pts_proj_rc.append(pts_proj) 

        # project on the left primary camera
        tf_cam2tt = ret["lpc2tt"] 
        tf_cam2target = np.linalg.inv(tf_target2tt_i) @ tf_cam2tt 
        tf_cam2tt_init = initials["lpc2tt"] 
        pts_2d, pts_proj, img_lpc = evaluate_reproj(img_lpc, pts_lpc["3d"], pts_lpc["2d"], tf_cam2target, tf_cam2tt, tf_cam2tt_init, \
                                                    initials["intrinsic_lpc"], tt_tf, track_tf, tt_tf_meas, track_tf_meas, debug)
        lpc_recorder.add_frame(img_lpc, text=f"frame-{idx}")
        pts_2d_lpc.append(pts_2d)
        pts_proj_lpc.append(pts_proj) 

        # project on the right primary camera
        tf_cam2tt = ret["rpc2tt"] 
        tf_cam2target = np.linalg.inv(tf_target2tt_i) @ tf_cam2tt 
        tf_cam2tt_init = initials["rpc2tt"] 
        pts_2d, pts_proj, img_rpc = evaluate_reproj(img_rpc, pts_rpc["3d"], pts_rpc["2d"], tf_cam2target, tf_cam2tt, tf_cam2tt_init, \
                                                    initials["intrinsic_rpc"], tt_tf, track_tf, tt_tf_meas, track_tf_meas, debug)
        rpc_recorder.add_frame(img_rpc, text=f"frame-{idx}")
        pts_2d_rpc.append(pts_2d)
        pts_proj_rpc.append(pts_proj) 

    print("robot camera statistics:")
    print_reproj_stats(pts_2d_rc, pts_proj_rc)
    print("left primary camera statistics:")
    print_reproj_stats(pts_2d_lpc, pts_proj_lpc)
    print("right primary camera statistics:")
    print_reproj_stats(pts_2d_rpc, pts_proj_rpc)

    # dump result
    # if not debug:
    #     with open(".base2track.yaml", "w+") as f:
    #         yaml.safe_dump({
    #             "transformation": ret.tolist()
    #         }, f, default_flow_style=False)
    #     with open(".track2tt.yaml", "w+") as f:
    #         yaml.safe_dump({
    #             "transformation": tf_track2tt.tolist()
    #         }, f, default_flow_style=False)

def evaluate_reproj(img_vis, pts_3d, pts_2d, tf_cam2target, tf_cam2tt, tf_cam2tt_init, intrinsic, tt_tf, track_tf, tt_meas, track_meas, debug):
    pts_proj = transfer_3d_pts_to_img(pts_3d, tf_cam2target, intrinsic)
    img_vis = visualize_reprojection(img_vis, pts_2d, pts_proj) 
    img_vis = turntable_projection(img_vis, tf_cam2tt, intrinsic, color=(0, 255, 0))
    # tf_cam2tt_init = initials["track2tt"] @ track_tf_meas @ initials["base2track"] @ ee2base_meas @ initials["cam2ee"] 
    img_vis = turntable_projection(img_vis, tf_cam2tt_init, intrinsic, color=(0, 0, 255), show_axes=False)
    img_vis = cv.resize(img_vis, (int(img_vis.shape[1]/2), int(img_vis.shape[0]/2)))
    cv.putText(
        img_vis, 
        f"track_diff: {tf_mat_diff(track_meas, track_tf)} tt_diff: {tf_mat_diff(tt_meas, tt_tf)}",
        (50, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0)
    )
    if debug:
        cv.imshow("vis", img_vis)
        cv.waitKey(0)
    return pts_2d, pts_proj, img_vis


def turntable_projection(img, tf_cam2tt, intrinsic, color=(0, 255, 0), tt_radius=2.75, show_axes=True):
    # make virtual edge points of turntable
    pts_tt = []
    for angle in np.linspace(0, 2 * np.pi, 100):
        angle = wrap_around_rad(angle)
        x = tt_radius * np.cos(angle)
        y = tt_radius * np.sin(angle)
        pts_tt.append([x, y, 0])
    pts_tt.append([0, 0, 0])
    pts_tt = np.array(pts_tt)
    pts_proj = transfer_3d_pts_to_img(pts_tt, tf_cam2tt, intrinsic)
    # draw arrows
    pts_axis = np.eye(3) 
    pts_axis_proj = transfer_3d_pts_to_img(pts_axis, tf_cam2tt, intrinsic)
    img = draw_pts_on_img(img, pts_proj, s=3, c=color)
    if show_axes:
        axis_color = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0)
        ]
        axis_names = ["x", "y", "z"]
        for a_c, a_n, pt in zip(axis_color, axis_names, pts_axis_proj.astype(int)):
            cv.arrowedLine(img, tuple(pts_proj[-1].astype(int)), tuple(pt), a_c, 2, cv.LINE_AA, 0, 0.2)
            text_loc = pt.copy() 
            text_loc[0] += 20
            cv.putText(
                img, 
                a_n,
                tuple(text_loc.astype(int)),
                cv.FONT_HERSHEY_SIMPLEX,
                1.2,
                a_c 
            )
    return img 

def lidar2cam_proj(pts_lidar, img, tf_cam2lidar, tf_cam2lidar_init, intrinsic, debug=False):
    pts_img = transfer_3d_pts_to_img(pts_lidar, tf_cam2lidar, intrinsic)
    img_vis = draw_pts_on_img(img, pts_img)
    pts_img = transfer_3d_pts_to_img(pts_lidar, tf_cam2lidar_init, intrinsic)
    img_vis = draw_pts_on_img(img_vis, pts_img, c=(0, 0, 255))
    img_vis = cv.resize(img_vis, (int(img_vis.shape[1]/2), int(img_vis.shape[0]/2)))
    if debug:
        cv.imshow("vis", img_vis)
        cv.waitKey(0)
    return img_vis

def calibrate_joint_calib_w_lidar(bag_path=None, perturb=False, debug=False):
    track_tfs = []
    tt_tfs = []
    pts_rc_all, pts_lpc_all, pts_rpc_all = [], [], []
    lidar_features = []
    initials = get_gt()

    rc_recorder = VideoRecorder(2, filename="rc.mp4")
    lpc_recorder = VideoRecorder(2, filename="lpc.mp4")
    rpc_recorder = VideoRecorder(2, filename="rpc.mp4")
    lidar_recorder = VideoRecorder(2, filename="lidar.mp4")

    for _, (_, _, _, _, pts_rc, pts_lpc, pts_rpc, lidar_feature, hand_pose, track_tf, tt_tf) in enumerate(get_data(bag_path)):
        track_tfs.append(track_tf)
        tt_tfs.append(tt_tf)
        pts_rc_all.append(pts_rc)
        pts_lpc_all.append(pts_lpc)
        pts_rpc_all.append(pts_rpc)
        lidar_features.append(lidar_feature)
    initials["ee2base"] = hand_pose # note that the hand is not moving during calibration

    # perturbation test
    if perturb:
        pert = deepcopy(initials)
        pert["track2tt"] = perturb_pose3(pert["track2tt"], [1e-3, 1e-3, 0, 0.1, 0.1, 1e-2]) 
        pert["base2track"] = perturb_pose3(pert["base2track"], [1e-5, 1e-5, 0.1, 0, 0, 0]) 
        track_tfs = [perturb_pose3(tr_tf, [0, 0, 0, 1e-2, 0, 0]) for tr_tf in track_tfs]
        tt_tfs = [perturb_pose3(tt_tf, [0, 0, 2e-3, 0, 0, 0]) for tt_tf in tt_tfs]
        pert["ee2base"] = perturb_pose3(initials["ee2base"], [1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4])
        pts_rc_all = [{"2d": perturb_pts(pts["2d"], [1, 1]), "3d": pts["3d"]} for pts in pts_rc_all] 
        pts_lpc_all = [{"2d": perturb_pts(pts["2d"], [1, 1]), "3d": pts["3d"]} for pts in pts_lpc_all] 
        pts_rpc_all = [{"2d": perturb_pts(pts["2d"], [1, 1]), "3d": pts["3d"]} for pts in pts_rpc_all] 
        pert["target2tt"] = perturb_pose3(pert["track2tt"], [1e-2, 1e-5, 1e-5, 1e-1, 1e-4, 1e-1])
    else:
        pert = initials

    print("frontend features collection finished, running optimizatin...")
    ret = solve_joint_calib_2(pts_rc_all, pts_lpc_all, pts_rpc_all, lidar_features, track_tfs, tt_tfs, pert)
    for key, val in ret.items():
        if "2" in key: # only the static transforms
            print(f"diff of {key}: ", mat_to_euler_vec(np.linalg.inv(initials[key]) @ val))
            # print(f"diff of {key}: ", mat_to_euler_vec(initials[key] @ np.linalg.inv(val)))
    print("track2tt: ", mat_to_euler_vec(ret["track2tt"]))
    print("base2track: ", mat_to_euler_vec(ret["base2track"]))

    # evaluate the result
    pts_2d_rc, pts_proj_rc = [], [] 
    pts_2d_lpc, pts_proj_lpc = [], []
    pts_2d_rpc, pts_proj_rpc = [], []
    for idx, ((img_rc, img_lpc, img_rpc, pts_lidar, pts_rc, pts_lpc, pts_rpc, _, ee2base_meas, track_tf_meas, tt_tf_meas), track_tf, tt_tf) in enumerate(zip(get_data(bag_path), ret["track_tfs"], ret["tt_tfs"])):
        tf_target2tt_i = tt_tf @ ret["target2tt_0"] 

        # project on the robot camera
        tf_cam2tt = ret["track2tt"] @ track_tf @ ret["base2track"] @ ret["ee2base"] @ initials["cam2ee"]
        tf_cam2target = np.linalg.inv(tf_target2tt_i) @ tf_cam2tt
        tf_cam2tt_init = initials["track2tt"] @ track_tf @ initials["base2track"] @ ret["ee2base"] @ initials["cam2ee"] 
        pts_2d, pts_proj, img_rc = evaluate_reproj(img_rc, pts_rc["3d"], pts_rc["2d"], tf_cam2target, tf_cam2tt, tf_cam2tt_init, \
                                                   initials["intrinsic"], tt_tf, track_tf, tt_tf_meas, track_tf_meas, debug)
        rc_recorder.add_frame(img_rc, text=f"frame-{idx}")
        pts_2d_rc.append(pts_2d)
        pts_proj_rc.append(pts_proj) 

        # project on the left primary camera
        tf_cam2tt = ret["lpc2tt"] 
        tf_cam2target = np.linalg.inv(tf_target2tt_i) @ tf_cam2tt 
        tf_cam2tt_init = initials["lpc2tt"] 
        pts_2d, pts_proj, img_lpc_vis = evaluate_reproj(img_lpc, pts_lpc["3d"], pts_lpc["2d"], tf_cam2target, tf_cam2tt, tf_cam2tt_init, \
                                                    initials["intrinsic_lpc"], tt_tf, track_tf, tt_tf_meas, track_tf_meas, debug)
        lpc_recorder.add_frame(img_lpc_vis, text=f"frame-{idx}")
        pts_2d_lpc.append(pts_2d)
        pts_proj_lpc.append(pts_proj) 

        # project on the right primary camera
        tf_cam2tt = ret["rpc2tt"] 
        tf_cam2target = np.linalg.inv(tf_target2tt_i) @ tf_cam2tt 
        tf_cam2tt_init = initials["rpc2tt"] 
        pts_2d, pts_proj, img_rpc = evaluate_reproj(img_rpc, pts_rpc["3d"], pts_rpc["2d"], tf_cam2target, tf_cam2tt, tf_cam2tt_init, \
                                                    initials["intrinsic_rpc"], tt_tf, track_tf, tt_tf_meas, track_tf_meas, debug)
        rpc_recorder.add_frame(img_rpc, text=f"frame-{idx}")
        pts_2d_rpc.append(pts_2d)
        pts_proj_rpc.append(pts_proj) 

        # add lidar projection on one of the primary cameras
        tf_lidar2cam = np.linalg.inv(ret["lidar2tt"]) @ ret["lpc2tt"]
        tf_lidar2cam_init = np.linalg.inv(initials["lidar2tt"]) @ initials["lpc2tt"]
        img_lidar = lidar2cam_proj(pts_lidar, img_lpc, tf_lidar2cam, tf_lidar2cam_init, initials["intrinsic_lpc"], debug=debug)
        lidar_recorder.add_frame(img_lidar, text=f"frame-{idx}")

    print("robot camera statistics:")
    print_reproj_stats(pts_2d_rc, pts_proj_rc)
    print("left primary camera statistics:")
    print_reproj_stats(pts_2d_lpc, pts_proj_lpc)
    print("right primary camera statistics:")
    print_reproj_stats(pts_2d_rpc, pts_proj_rpc)

if __name__ == "__main__":
    # calibrate_joint_calib(bag_path=None, perturb=True, debug=True)
    # calibrate_joint_calib(bag_path="/home/fuhengdeng/test_data/base_tt.bag", perturb=False, debug=False)
    calibrate_joint_calib_w_lidar(bag_path="/home/fuhengdeng/test_data/base_tt.bag", perturb=False, debug=False)