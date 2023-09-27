import numpy as np
from abc import ABC, abstractclassmethod

from py_kinetic_backend import Rot3, Pose3


class CalibTarget3D(ABC):
    def __init__(self):
        pass

    @abstractclassmethod
    def get_pts_3d():
        pass


class CheckerBoardTarget(CalibTarget3D):
    def __init__(self, rows: int, cols: int, checker_size: float):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.checker_size = checker_size 

    def get_pts_3d(self):
        points = []
        for i in range(self.rows):
            for j in range(self.cols):
                x = j * self.checker_size
                y = i * self.checker_size
                z = 0  # The points lie on the z=0 plane in the target frame
                points.append((x, y, z))
        # make (0, 0) to be the checkerboard center
        points = np.array(points)
        points -= points.max(axis=0) / 2 
        return points 

class ArucoBoardTarget(CalibTarget3D):
    def __init__(self, rows, cols, marker_size, sep_size, start_id):
        self.rows = rows
        self.cols = cols
        self.marker_size = marker_size
        self.sep_size = sep_size
        self.start_id = start_id  # the start marker is defined as the left upmost corner
        self._pts_3d = self.make_pts_3d()
        super().__init__()

    def make_pts_3d(self):
        points = []
        for i in range(self.rows):
            for j in range(self.cols):
                square = self.make_square_clockwise(
                    j * (self.marker_size + self.sep_size),
                    i * (self.marker_size + self.sep_size),
                    0
                )
                points.append(square)
        points = np.vstack(points)
        points -= points.max(axis=0) / 2 
        return points

    def get_pts_3d(self):
        return self._pts_3d

    def make_square_clockwise(self, x_ul, y_ul, z_ul):
        return np.array([
            [x_ul, y_ul, z_ul],
            [x_ul + self.marker_size, y_ul, z_ul],
            [x_ul + self.marker_size, y_ul + self.marker_size, z_ul],
            [x_ul, y_ul + self.marker_size, z_ul],
        ])

    def find_3d_pts_by_id(self, id):
        id_local = id - self.start_id
        if id_local < 0 or id_local >= self.rows * self.cols:
            return None
        return self._pts_3d[id_local*4:(id_local+1)*4, :]


class ArucoCubeTarget(CalibTarget3D):
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

    def get_pts_3d_by_id(self, id):
        frame = self._frames[id]
        pose = Pose3(Rot3.rzryrx(frame[:3]), frame[3:])
        pts_target = []
        for pts in self._targets[id].get_pts_3d():
            pts_target.append(pose.transform_to(pts))
        return np.array(pts_target) 

    def get_pts_3d(self):
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
            frame = self._frames[key]
            pose = Pose3(Rot3.rzryrx(frame[:3]), frame[3:])
            pts_ret = []
            for pts in ret:
                pts_ret.append(pose.transform_to(pts))
            break
        else:
            return None 
        return pts_ret

           