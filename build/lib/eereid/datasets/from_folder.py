from eereid.datasets.dataset import dataset
import numpy as np

import re
import os
from glob import glob

from PIL import Image

from tqdm import tqdm

class from_folder(dataset):
    def __init__(self,pth,label,include=None):
        if include is None:
            include= lambda x: True
        self.pth = pth
        self.label = label
        self.include = include
        super().__init__("from_folder")
        self.search_files()

    def search_files(self):
        if type(self.pth) is list:
            files = []
            for p in self.pth:
                files += glob(p)
        else:
            files=glob(self.pth)
        if callable(self.include):
            files = [f for f in files if self.include(f)]
        elif type(self.include) is str:
            files = [f for f in files if re.match(self.include,f)]
        else:
            raise ValueError("include must be a callable or a string")
        files.sort()
        self.files = files

        print("Found {} files".format(len(self.files)))

        

    def load_raw(self):
        x,y=[],[]
        itera=tqdm(self.files)
        itera.set_description("Loading images")
        for fn in itera:
            x.append(np.array(Image.open(fn)))
        if callable(self.label):
            y = [self.label(f) for f in self.files]
        elif type(self.label) is str:
            y = [re.match(self.label,f).group(1) for f in self.files]
        else:
            raise ValueError("label must be a callable or a string")
        x = np.array(x)
        y = np.array(y)
        return x,y

    def input_shape(self):
        return np.array(Image.open(self.files[0])).shape
    
    def sample_count(self):
        return len(self.files)

    def save(self,pth):
        super().save(pth,pth=self.pth,label=self.label,include=self.include)



