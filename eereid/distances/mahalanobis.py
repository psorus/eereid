from eereid.distances.distance import distance

import numpy as np


class mahalanobis(distance):
    def __init__(self):
        super().__init__("mahalanobis")

    def distance(self,a,b):
        raise Exception("Not implemented(able)")

    def multi_distance(self,A,b):
        cov = np.cov(A.T)
        inv_cov = np.linalg.inv(cov)
        return np.sqrt(np.diag(np.dot(np.dot((A-b),inv_cov),(A-b).T)))



