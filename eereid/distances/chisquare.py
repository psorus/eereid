from eereid.distances.distance import distance

import numpy as np


class chisquare(distance):
    def __init__(self):
        super().__init__("chisquare")

    def distance(self,a,b):
        return np.sum((a-b)**2/(a+b))

    def multi_distance(self,A,b):
        return np.sum((A-b)**2/(A+b),axis=1)



