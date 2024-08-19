from eereid.prepros.prepro import prepro

import numpy as np

class featurebagging(prepro):
    def __init__(self,feats):
        self.feats=feats
        super().__init__("featurebagging")

    def apply(self,data, labels, eereid):
        if eereid is not None:self._apply_special(eereid)
        data=np.reshape(data, (data.shape[0],-1))
        return data[:,self.selected], labels

    def _apply_one(self, image):
        image=np.reshape(image, -1)
        return image[self.selected]

    def _apply_special(self,eereid):
        feats=self.feats
        inputs=np.prod(eereid.input_shape)
        if type(feats)==float:feats=int(feats*inputs)
        self.selected=np.random.choice(inputs,feats,replace=False)
        eereid.input_shape=[feats]

    def save(self,pth,index):
        super().save(pth,index,size=size)

    def stage(self):return "general"
    def order(self):return 20

    def explain(self):
        return f"Flattening each input data into a vector"
