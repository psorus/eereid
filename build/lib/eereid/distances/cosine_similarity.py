from eereid.distances.distance import distance

import numpy as np


class cosine_similarity(distance):#prob buggy rn
    def __init__(self):
        super().__init__("cosine_similarity")

    def distance(self,a,b):
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def multi_distance(self,A,b):
        return np.dot(A,b)/(np.linalg.norm(A, axis=1)*np.linalg.norm(b))

    def explain(self):
        return "Cosine similarity is a measure of similarity between two vectors. It is calculated as the dot product of the two vectors divided by the product of their magnitudes. The formula is: dot(a,b)/(norm(a)*norm(b))"



