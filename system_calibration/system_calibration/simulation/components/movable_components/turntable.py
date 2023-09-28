from .movable_component import MovableComponent 
from ....utils import euler_vec_to_mat, wrap_around_rad

class Turntable(MovableComponent):
    def __init__(self):
        super().__init__()

    def move(self, yaw_rad: float):
        if not isinstance(yaw_rad, float) and not isinstance(yaw_rad, int): 
            raise ValueError("Turntable can only move in 1 dim, please input float") 
        # make sure yaw rad is always between [-pi, pi]
        yaw_rad = wrap_around_rad(-yaw_rad) 
        self._frame = euler_vec_to_mat([0, 0, yaw_rad, 0, 0, 0], use_deg=False)

