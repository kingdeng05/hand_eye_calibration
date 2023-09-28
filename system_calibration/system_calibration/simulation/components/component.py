import numpy as np
from abc import ABC 


class Component(ABC):
    def __init__(self):
        self._frame = np.eye(4)

    @property
    def frame(self):
        return self._frame 