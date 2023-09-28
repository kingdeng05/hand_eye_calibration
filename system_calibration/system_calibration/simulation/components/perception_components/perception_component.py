from abc import abstractmethod

from ..component import Component


class PerceptionComponent(Component):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def capture(self, target):
        pass 
    