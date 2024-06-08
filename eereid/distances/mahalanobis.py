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

    def explain(self):
        return "Mahalanobis distance is a measure of distance between two vectors. It is calculated as the square root of the sum of the squared differences between the two vectors, where the differences are scaled by the inverse of the covariance matrix. The formula is: sqrt((a-b)*inv(cov)*(a-b)^T)"


