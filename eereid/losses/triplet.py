from eereid.losses.loss import loss

import numpy as np

from tensorflow.keras import backend as K

class triplet(loss):
    def __init__(self, margin=1.0):
        self.margin = margin
        super().__init__("triplet")

    def build(self,mods):

        def func(y_true, y_pred):
            #print(y_true.shape,y_pred.shape)
            #exit()
            anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
            positive_dist = K.sum(K.square(anchor - positive), axis=-1)
            negative_dist = K.sum(K.square(anchor - negative), axis=-1)
            return K.sum(K.maximum(positive_dist - negative_dist + self.margin, 0), axis=-1)

        return func

    def save(self,pth):
        super().save(pth,margin=self.margin)

    def Nlet_string(self):
        return "aab"
        

