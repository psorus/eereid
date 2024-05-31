import numpy as np
from eereid.gag import gag

class experimental(gag):
    def __init__(self,name):
        super().__init__(name)

    def save(self,pth,index,**kwargs):
        self.savehelper2(pth,"experimental",index,**kwargs)

    def species(self):return "experimental"

    def ident(self):
        raise NotImplementedError("This is an abstract method")
