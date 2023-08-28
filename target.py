import numpy as np

from abc import ABC, abstractclassmethod


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
                x = i * self.checker_size
                y = j * self.checker_size
                z = 0  # The points lie on the z=0 plane in the target frame
                points.append((x, y, z))
        # make (0, 0) to be the checkerboard center
        points = np.array(points)
        points -= points.max(axis=0) / 2 
        return np.array(points)
