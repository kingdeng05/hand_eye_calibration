import numpy as np

from .target import Target

class CheckerboardTarget(Target):
    def __init__(self, rows: int, cols: int, checker_size: float):
        self.rows = rows
        self.cols = cols
        self.checker_size = checker_size 
        super().__init__()

    def get_pts(self):
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
