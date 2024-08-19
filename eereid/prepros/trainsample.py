from eereid.prepros.prepro import prepro
import numpy as np

class trainsample(prepro):
    def __init__(self,frac=0.1):
        super().__init__("trainsample")
        self.frac=frac

    def apply(self, data, labels, eereid):
        classes=list(set(labels))
        np.random.shuffle(classes)
        count=len(classes)
        allowed=int(np.ceil(count*self.frac))
        ac=classes[:allowed]

        datas=np.array([data for data,label in zip(data,labels) if label in ac])
        labels=np.array([label for label in labels if label in ac])

        return datas,labels

    def save(self,pth,index):
        super().save(pth,index,frac=self.frac,seed=self.seed)

    def stage(self):return "train"
    def order(self):return 19

    def explain(self):
        return f"Reduces the fraction of samples used by factor {self.frac}. This is mostly useful for quickly debugging. In comparison to subsample, this method keep every image of every class taken, but only limits the number of classes. This also increases the accuracy"
