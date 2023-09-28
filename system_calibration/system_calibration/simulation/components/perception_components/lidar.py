import numpy as np

from .perception_component import PerceptionComponent

class LiDAR(PerceptionComponent):
    def __init__(self):
        super().__init__()

    def capture(self):
        print("To be implemented") 
        return np.array([]).reshape(-1, 3)
        