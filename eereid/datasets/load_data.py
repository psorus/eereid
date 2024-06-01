from eereid.datasets.dataset import dataset
import numpy as np

import functools


class load_data(dataset):
    def __init__(self,pth):
        super().__init__("load_data")
        self.pth = pth

    @functools.cache
    def load_raw(self):
        f=np.load(self.pth)
        return f['x'],f['y']

    def load_data(self,mods=None):
        return self.load_raw()

    def input_shape(self):
        return self.load_raw()[0].shape[1:]
    
    def sample_count(self):
        return len(self.load_raw()[0])



