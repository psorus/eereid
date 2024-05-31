from eereid.distances.distance import distance

import numpy as np


class custom_distance(distance):
    def __init__(self,func):
        super().__init__("custom_distance")
        self.func = func

    def distance(self,a,b):
        return self.func(a,b)

    def save(self,pth):
        raise NotImplementedError

