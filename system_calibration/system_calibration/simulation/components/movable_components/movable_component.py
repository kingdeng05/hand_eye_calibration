import numpy as np
from abc import abstractmethod

from ..component import Component


class MovableComponent(Component):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def move(self, val):
        pass

