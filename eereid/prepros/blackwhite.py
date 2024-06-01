from eereid.prepros.prepro import prepro
import numpy as np

class blackwhite(prepro):
    def __init__(self, r=0.299, g=0.587, b=0.114):
        sm=r+g+b
        r=r/sm
        g=g/sm
        b=b/sm
        self.vec=np.array([r,g,b])
        self.r = r
        self.g = g
        self.b = b
        super().__init__("blackwhite")

    def apply(self, data, labels, eereid):
        data=np.dot(data,self.vec)
        self._apply_special(eereid)
        return data, labels

    def _apply_one(self, image):
        return np.dot(image, self.vec)

    def _apply_special(self,eereid):
        eereid.input_shape=eereid.input_shape[:-1]

    def save(self,pth,index):
        super().save(pth,index,r=self.r,g=self.g,b=self.b)

    def stage(self):return "general"
    def order(self):return 3

