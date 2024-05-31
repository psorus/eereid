from eereid.losses.loss import loss

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

class quadruplet(loss):
    def __init__(self, margin=1.0):
        self.margin = margin
        super().__init__("quadruplet")

    def build(self,mods):

        typ=mods("loss_aggregator","avg")

        def func(y_true, y_pred):
            anchor, positive, negative, negative2 = y_pred[0], y_pred[1], y_pred[2], y_pred[3]
            positive_dist = K.sum(K.square(anchor - positive), axis=-1)
            negative_dist = K.sum(K.square(anchor - negative), axis=-1)
            negative_dist2= K.sum(K.square(anchor - negative2), axis=-1)
            if typ=="min":
                return K.sum(K.maximum(positive_dist - K.minimum(negative_dist,negative_dist2)  + self.margin, 0), axis=-1)
            elif typ=="max":
                return K.sum(K.maximum(positive_dist - K.maximum(negative_dist,negative_dist2)  + self.margin, 0), axis=-1)
            elif typ=="avg":
                return K.sum(K.maximum(positive_dist - (negative_dist+negative_dist2)/2  + self.margin, 0), axis=-1)
            elif typ=="sum":
                return K.sum(K.maximum(positive_dist - (negative_dist+negative_dist2)  + self.margin, 0), axis=-1)
            else:
                raise ValueError("Invalid type",typ)

        return func

    def save(self,pth):
        super().save(pth,margin=self.margin)
        
    def Nlet_string(self):
        return "aabc"

