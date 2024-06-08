import numpy as np

from eereid.gag import gag

import os
import json


class dataset(gag):
    def __init__(self,name,seed=42):
        super().__init__(name)
        self.seed=12

    def load_raw(self):
        """returns samples and labels"""
        raise NotImplementedError

    def load_data(self,mods=None):
        data,labels=self.load_raw()
        rng=np.random.RandomState(self.seed)
        indices=np.arange(len(data))
        rng.shuffle(indices)
        data=data[indices]
        labels=labels[indices]
        return data,labels

    def save(self,pth,**kwargs):
        self.savehelper(pth,"dataset.json",seed=self.seed,**kwargs)

    def input_shape(self):
        raise NotImplementedError

    def sample_count(self):
        raise NotImplementedError

    def species(self):return "dataset"

    def __add__(self,other):
        if type(self) is mergeds and type(other) is mergeds:
            return mergeds(*self.datasets,*other.datasets,seed=self.seed)
        if type(self) is mergeds:
            return self.add_dataset(other)
        if type(other) is mergeds:
            return other.add_dataset(self)
        return mergeds(self,other)

    def explain(self):return "Generic data loader gag"

class mergeds(dataset):
    def __init__(self,*datasets,seed=42):
        super().__init__("mergeds")
        self.datasets=list(datasets)
        self.seed=seed
        
    def add_dataset(self,dataset): 
        self.datasets.append(dataset)
        return self

    def load_raw(self):
        x,y=[],[]
        for d in self.datasets:
            xx,yy=d.load_data()
            x.append(xx)
            y.append(yy)
        x=np.concatenate(x)
        y=np.concatenate(y)
        rnd=np.random.RandomState(self.seed)
        idx=rnd.permutation(x.shape[0])
        x=x[idx]
        y=y[idx]
        return x,y

    def input_shape(self):
        return self.datasets[0].input_shape()
    
    def sample_count(self):
        return np.sum([d.sample_count() for d in self.datasets])

    def save(self,pth,index):
        raise NotImplementedError

    def explain(self):
        return "Combination of multiple datasets:\n"+\
            "\n".join(["    "+d.explain() for d in self.datasets])



