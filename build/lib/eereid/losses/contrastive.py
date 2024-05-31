from eereid.losses.loss import loss

import numpy as np

from tensorflow.keras import backend as K

class contrastive(loss):
    def __init__(self, margin=1.0):
        self.margin = margin
        super().__init__("contrastive")

    def build(self,mods):

        def func(y_true, y_pred):
            #aa: 1/2 D**2
            #ab: 1/2 max(0,margin-D)**2
            anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
            positive_dist = K.sum(K.square(anchor - positive), axis=-1)
            negative_dist = K.sum(K.square(anchor - negative), axis=-1)
            contrast=(positive_dist+K.maximum(0.,self.margin-negative_dist))
            return K.sum(contrast, axis=-1)

        return func

    def save(self,pth):
        super().save(pth,margin=self.margin)

    def Nlet_string(self):
        #usual contrastive would only use aa/ab, but this requires a y value. So for consistency we absorb both terms into one aab loss function
        return "aab"
        

