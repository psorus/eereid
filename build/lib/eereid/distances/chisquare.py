from eereid.distances.distance import distance

import numpy as np


class chisquare(distance):
    def __init__(self):
        super().__init__("chisquare")

    def distance(self,a,b):
        return np.sum((a-b)**2/(a+b))

    def multi_distance(self,A,b):
        return np.sum((A-b)**2/(A+b),axis=1)

    def explain(self):
        return "Chi-Square distance function (chisquare) is a measure of distance between two vectors. It is calculated as the sum of the squared differences between the two vectors divided by the sum of the two vectors. The formula is: sum((a-b)^2/(a+b))"


