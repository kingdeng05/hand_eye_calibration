import numpy as np
import cv2 as cv

from copy import deepcopy


class IntrinsicCailbrator(object):
    def __init__(self, iteration=3, outlier_perc=99, flags=cv.CALIB_RATIONAL_MODEL, use_auto_stop=True):
        self._iteration = iteration 
        self._outlier_perc = outlier_perc
        self._flags = flags 
        self._min_view_pts = 3
        self._auto_stop_ratio_prev = None 
        self._auto_stop_thres = 0.05
        self._use_auto_stop = use_auto_stop

    def calibrate(self, pts_3d, pts_2d, img_size):
        pts_3d_calib = deepcopy(pts_3d)
        pts_2d_calib = deepcopy(pts_2d)
        for idx in range(self._iteration):
            print(f"Calibrating intrinsic iter {idx+1} with {len(pts_3d_calib)} views...")
            _, k, d, rvecs, tvecs = cv.calibrateCamera(pts_3d_calib, pts_2d_calib, img_size, None, None, flags=self._flags)
            print("Done calibration & filtering outliers...")
            error_views = self.compute_reproj_error(k, d, rvecs, tvecs, pts_3d_calib, pts_2d_calib)
            outlier_thres = np.percentile(np.hstack(error_views), self._outlier_perc)
            max_error = np.hstack(error_views).max()
            auto_stop_ratio = max_error / outlier_thres
            print(f"Outlier {self._outlier_perc}% threshold computed: {outlier_thres}, error max: {max_error}, auto stop ratio: {auto_stop_ratio}")
            if self._use_auto_stop and \
               self._auto_stop_ratio_prev is not None and \
               np.abs(auto_stop_ratio - self._auto_stop_ratio_prev) < self._auto_stop_thres: 
                print(f"Ratio {auto_stop_ratio} has reached within threshold of previous ratio {self._auto_stop_ratio_prev}, exiting...")
                break
            self._auto_stop_ratio_prev = auto_stop_ratio 
            pts_3d_calib, pts_2d_calib = self.filter_outliers(pts_3d_calib, pts_2d_calib, error_views, outlier_thres)
        return k, d, rvecs, tvecs, pts_3d_calib, pts_2d_calib 

    @staticmethod
    def compute_reproj_error(k, d, rvecs, tvecs, pts_3d, pts_2d):
        error_views = []
        for pts_3d_view, pts_2d_view, rvec, tvec in zip(pts_3d, pts_2d, rvecs, tvecs):
            pts_proj_view, _ = cv.projectPoints(pts_3d_view, rvec, tvec, k, d)
            error = np.linalg.norm(pts_2d_view.reshape(-1, 2) - pts_proj_view.reshape(-1, 2), axis=1)
            error_views.append(error)
        return error_views 

    def filter_outliers(self, pts_3d, pts_2d, error_views, outlier_thres):
        pts_3d_good = []
        pts_2d_good = []
        for pts_3d_view, pts_2d_view, error_view in zip(pts_3d, pts_2d, error_views):
            assert(len(pts_3d_view) == len(pts_2d_view) == len(error_view))
            good_idx = error_view < outlier_thres
            # only accepting the views that has more than minimum view points
            if np.sum(good_idx) >= self._min_view_pts:
                pts_3d_good.append(pts_3d_view[good_idx])
                pts_2d_good.append(pts_2d_view[good_idx])
        return pts_3d_good, pts_2d_good 

