import numpy as np

from .target import Target

class ArucoBoardTarget(Target):
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

    def get_pts(self):
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