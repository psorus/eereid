from eereid.distances.distance import distance

import numpy as np


class manhattan(distance):
    def __init__(self):
        super().__init__("manhattan")

    def distance(self,a,b):
        return np.linalg.norm(a-b, ord=1)

    def multi_distance(self,A,b):
        return np.linalg.norm(A-b, ord=1, axis=1)

    def explain(self):
        return "Manhattan distance is a measure of distance between two vectors. It is calculated as the sum of the absolute differences between the two vectors. The formula is: sum(abs(a-b))"



