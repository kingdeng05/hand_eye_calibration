import numpy as np

from .movable_component import MovableComponent 
from ....utils import euler_vec_to_mat

class Robot(MovableComponent):
    def __init__(self):
        super().__init__()

    def move(self, vec, use_deg=False):
        if not isinstance(vec, list) and not isinstance(vec, np.ndarray): 
            raise ValueError("Please input 6dim vec for robot motion") 
        self._frame = euler_vec_to_mat(vec, use_deg=use_deg)
