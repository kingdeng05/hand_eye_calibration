import numpy as np

from .target import Target
from .aruco_cube_target import ArucoCubeTarget 
from .aruco_board_target import ArucoBoardTarget 

class ArucoCubeTargetV2(ArucoCubeTarget):
    def __init__(self, size, use_ids=None):
        super().__init__(size)
        targets = { 
            0: ArucoBoardTarget(3, 3, 0.166, 0.033, 0),
            9: ArucoBoardTarget(3, 3, 0.166, 0.033, 9),
            18: ArucoBoardTarget(3, 3, 0.166, 0.033, 18),
            27: ArucoBoardTarget(3, 3, 0.166, 0.033, 27),
            36: ArucoBoardTarget(3, 3, 0.166, 0.033, 36),
        } 
        frames = {
            0: np.array([np.pi/2, 0, -np.pi*3/4, 0, 0, size/2]), # A 
            9: np.array([0, np.pi*3/4, 0, 0, 0, size/2]), # B 
            18: np.array([0, -np.pi/4, 0, 0, 0, size/2]), # C 
            27: np.array([0, -np.pi*3/4, 0, 0, 0, size/2]), # D 
            36: np.array([0, np.pi/4, 0, 0, 0, size/2]) # E 
        } 

        # filter the targets and frames based on use_ids
        if use_ids is None:
            use_ids = list(targets.keys())
        self._targets = {use_id: targets[use_id] for use_id in use_ids}
        self._frames = {use_id: frames[use_id] for use_id in use_ids}
        self._use_ids = use_ids
