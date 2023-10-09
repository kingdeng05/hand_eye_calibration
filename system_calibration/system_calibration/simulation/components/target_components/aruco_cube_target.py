import numpy as np

from .target import Target
from .aruco_board_target import ArucoBoardTarget

from ....utils import euler_vec_to_mat, transform_3d_pts

class ArucoCubeTarget(Target):
    def __init__(self, size, use_ids=None):
        targets = { 
            0: ArucoBoardTarget(5, 5, 0.166, 0.033, 0),
            25: ArucoBoardTarget(5, 5, 0.166, 0.033, 25),
            50: ArucoBoardTarget(5, 5, 0.166, 0.033, 50),
            75: ArucoBoardTarget(5, 5, 0.166, 0.033, 75),
            100: ArucoBoardTarget(5, 5, 0.166, 0.033, 100),
        } 
        # rx, ry, rz, x, y, z (in euler angles), from base to side of target
        frames = {
            0: np.array([np.pi/2, 0, -np.pi*3/4, 0, 0, size/2]), # A 
            25: np.array([0, np.pi*3/4, 0, 0, 0, size/2]), # B 
            50: np.array([0, -np.pi/4, 0, 0, 0, size/2]), # C 
            75: np.array([0, -np.pi*3/4, 0, 0, 0, size/2]), # D 
            100: np.array([0, np.pi/4, 0, 0, 0, size/2]) # E 
        } 
        # filter the targets and frames based on use_ids
        if use_ids is None:
            use_ids = list(targets.keys())
        self._targets = {use_id: targets[use_id] for use_id in use_ids}
        self._frames = {use_id: frames[use_id] for use_id in use_ids}
        self._use_ids = use_ids
        super().__init__()

    @property
    def use_ids(self):
        return self._use_ids 

    @property
    def frames(self):
        return self._frames

    def get_pts_3d_by_id(self, id, to_base=True):
        tf_base2face = euler_vec_to_mat(self._frames[id], use_deg=False)
        tf_face2base = np.linalg.inv(tf_base2face)
        pts = self._targets[id].get_pts()
        if to_base: 
            pts = transform_3d_pts(pts, tf_face2base) 
        return pts 

    def get_pts(self):
        pts_all = []
        for use_id in self._use_ids:
            pts_target = self.get_pts_3d_by_id(use_id)
            pts_all.append(pts_target)
        pts_all = np.vstack(pts_all)
        return pts_all

    def find_3d_pts_by_id(self, id):
        for key, target in self._targets.items():
            ret = target.find_3d_pts_by_id(id)
            if ret is None:
                continue
            tf_base2face = euler_vec_to_mat(self._frames[id], use_deg=False)
            tf_face2base = np.linalg.inv(tf_base2face)
            pts_ret = transform_3d_pts(ret, tf_face2base) 
            break
        else:
            return None 
        return pts_ret

 