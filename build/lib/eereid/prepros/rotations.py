from eereid.prepros.prepro import prepro
import numpy as np

class rotations(prepro):
    def __init__(self,seed=42):
        super().__init__("rotations")
        self.seed=seed

    def apply(self, data, labels, eereid):
        datas=[]
        for i in range(4):
            datas.append(np.rot90(data,i,axes=(1,2)))
        datas=np.concatenate(datas,axis=0)
        labels=np.tile(labels,4)

        rnd=np.random.RandomState(self.seed)
        idx=rnd.permutation(datas.shape[0])
        datas=datas[idx]
        labels=labels[idx]

        return datas,labels

    def save(self,pth,index):
        super().save(pth,index,seed=self.seed)

