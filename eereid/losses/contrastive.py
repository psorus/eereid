from eereid.losses.loss import loss

import numpy as np

from tensorflow.keras import backend as K

class contrastive(loss):
    def __init__(self, margin=1.0):
        self.margin = margin
        super().__init__("contrastive")

    def build(self,mods):

        def func(y_true, y_pred):
            #print(y_true.shape,y_pred.shape)
            #exit()
            #aa: 1/2 D**2
            #ab: 1/2 max(0,margin-D)**2
            a,b=y_pred[0],y_pred[1]
            dist=K.sum(K.square(a-b),axis=-1)
            return K.sum(y_true*dist+(1-y_true)*K.maximum(0.,self.margin-dist),axis=-1)

        return func

    def save(self,pth):
        super().save(pth,margin=self.margin)

    def Nlet_string(self):
        #usual contrastive would only use aa/ab, but this requires a y value. So for consistency we absorb both terms into one aab loss function
        return "aa/ab"
        

