import numpy as np
from eereid.gag import gag

class distance(gag):
    def __init__(self,name):
        super().__init__(name)


    def distance(self,a,b):
        raise NotImplementedError

    def multi_distance(self,A,b):
        return np.array([self.distance(a,b) for a in A])


    def save(self,pth,**kwargs):
        self.savehelper(pth,"distance.json",**kwargs)

    def species(self):return "distance"

