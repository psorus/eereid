import numpy as np

from eereid.gag import gag

class prepro(gag):
    def __init__(self,name):
        super().__init__(name)

    def apply(self, data,labels,eereid):
        self._apply_special(eereid)
        return np.array([self._apply_one(d) for d in data]),labels


    def _apply_one(self, data):
        raise NotImplementedError

    def _apply_special(self,eereid):
        pass

    def save(self,pth,index,**kwargs):
        self.savehelper2(pth,"prepro",index,**kwargs)

    def species(self):return "prepro"

    def ident(self):
        return self.name
        #raise NotImplementedError("This is an abstract method")

