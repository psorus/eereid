from eereid.datasets.dataset import dataset
import numpy as np

from os.path import expanduser
import requests

import os

from tqdm import tqdm

hosts=["http://174.138.177.21:2156/eeri/{fn}"]
class downloadable(dataset):
    def __init__(self,name):
        super().__init__(name)

    def basepath(self):
        pth=expanduser("~/.eereid/data/")
        if not os.path.exists(pth):
            os.makedirs(pth)
        return pth
    def path(self):
        return self.basepath()+self.name+".npz"
    def download_file(self):
        path=self.path()
        for host in hosts:
            url=host.format(fn=self.name+".npz")
            print(f"Downloading {self.name} from {url}")
            r = requests.get(url, stream=True)
            total_size = int(r.headers.get('content-length', 0));
            block_size = 1024
            wrote = 0
            with open(path, 'wb') as f:
                for data in tqdm(r.iter_content(block_size), total=np.ceil(total_size//block_size), unit='KB', unit_scale=True):
                    wrote = wrote + len(data)
                    f.write(data)
            if total_size != 0 and wrote != total_size:
                print("ERROR, something went wrong")
                os.remove(path)
            else:
                return True
        return False
    def load_raw(self):
        if not os.path.exists(self.path()):
            assert self.download_file(), "Download failed"
        f=np.load(self.path())
        return f["x"],f["y"]


    def input_shape(self):
        return self.load_raw()[0].shape[1:]
    
    def sample_count(self):
        return len(self.load_raw()[0])



