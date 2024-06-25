from eereid.distances.distance import distance

import numpy as np


class euclidean(distance):
    def __init__(self):
        super().__init__("euclidean")

    def distance(self,a,b):
        return np.linalg.norm(a-b, ord=2)

    def multi_distance(self,A,b):
        return np.linalg.norm(A-b, ord=2, axis=1)

    def explain(self):
        return "Euclidean distance is a measure of distance between two vectors. It is calculated as the square root of the sum of the squared differences between the two vectors. The formula is: sqrt(sum((a-b)^2))"


