from eereid.datasets.dataset import dataset
import numpy as np


class from_numpy(dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
        super().__init__("from_numpy")

    def load_raw(self):
        return self.x,self.y

    def input_shape(self):
        return self.x.shape[1:]
    
    def sample_count(self):
        return len(self.x)



