from eereid.prepros.prepro import prepro
import numpy as np

class subsample(prepro):
    def __init__(self,frac=0.5,seed=42):
        super().__init__("subsample")
        self.frac=frac
        self.seed=seed

    def apply(self, data, labels, eereid):
        datas=[]

        rnd=np.random.RandomState(self.seed)

        use=round(data.shape[0]*self.frac)
        idx=rnd.permutation(data.shape[0])[:use]

        datas=data[idx]
        labels=labels[idx]

        return datas,labels

    def save(self,pth,index):
        super().save(pth,index,frac=self.frac,seed=self.seed)

    def stage(self):return "general"
    def order(self):return 0

    def explain(self):
        return f"Reduces the fraction of samples used by factor {self.frac}. This is mostly useful for quickly debugging"

    def apply_always(self):return False
