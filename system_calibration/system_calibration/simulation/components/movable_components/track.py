from .movable_component import MovableComponent 
from ....utils import euler_vec_to_mat

class Track(MovableComponent):
    def __init__(self):
        super().__init__()

    def move(self, x_val: float):
        if not isinstance(x_val, float) and not isinstance(x_val, int): 
            raise ValueError("Track can only move in 1 dim, please input float") 
        # track x direction is forward
        self._frame = euler_vec_to_mat([0, 0, 0, -x_val, 0, 0], use_deg=False)

