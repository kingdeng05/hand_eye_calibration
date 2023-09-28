from abc import abstractmethod

from ..component import Component

class Target(Component):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_pts():
        pass