from eereid.distances.distance import distance

import numpy as np


class lN(distance):
    def __init__(self, N):
        self.N = N
        super().__init__("lN")

    def distance(self,a,b):
        return np.linalg.norm(a-b, ord=self.N)

    def multi_distance(self,A,b):
        return np.linalg.norm(A-b, ord=self.N, axis=1)

    def save(self,pth):
        super.save(pth,N=self.N)

    def explain(self):
        return "lN distance is a measure of distance between two vectors. It is calculated as the Nth root of the sum of the Nth powers of the differences between the two vectors. The formula is: (sum((a-b)^N))^(1/N)"



