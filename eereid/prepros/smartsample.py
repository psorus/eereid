from eereid.prepros.prepro import prepro
import numpy as np

class smartsample(prepro):
    def __init__(self,frac=0.5):
        super().__init__("smartsample")
        self.frac=frac

    def apply(self, data, labels, eereid):
        classes=list(set(labels))
        class_to_data={c:[] for c in classes}
        for d,l in zip(data,labels):
            class_to_data[l].append(d)

        d,l=[],[]
        for c,vs in class_to_data.items():
            count=len(vs)*self.frac
            count=int(np.ceil(count))
            count=min(2,count)
            use=np.arange(len(vs))
            use=np.random.choice(use,count,replace=False)
            for u in use:
                d.append(vs[u])
                l.append(c)

        datas=np.array(d)
        labels=np.array(l)

        return datas,labels

    def save(self,pth,index):
        super().save(pth,index,frac=self.frac,seed=self.seed)

    def stage(self):return "general"
    def order(self):return 0

    def explain(self):
        return f"Reduces the fraction of samples used by factor {self.frac}. This is mostly useful for quickly debugging. In comparison to subsample, this method keeps the fraction of images taken per class constant."
    def apply_always(self):return False
