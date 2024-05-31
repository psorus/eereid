from eereid.losses.loss import loss

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

class ringloss(loss):
    def __init__(self, margin=10.0):
        self.margin = margin
        super().__init__("ringloss")

    def build(self,mods):

        typ=mods("loss_aggregator","avg")

        def func(y_true, y_pred):
            a,b,c=y_pred[0],y_pred[1],y_pred[2]
            d1=K.sum(K.square(a-b),axis=-1)
            d2=K.sum(K.square(b-c),axis=-1)
            d3=K.sum(K.square(c-a),axis=-1)
            aa=K.sum(K.square(a),axis=-1)
            bb=K.sum(K.square(b),axis=-1)
            cc=K.sum(K.square(c),axis=-1)

            loss=K.maximum(0.0,self.margin+aa+bb+cc-d1-d2-d3)

            return loss


        return func

    def save(self,pth):
        super().save(pth,margin=self.margin)
        
    def Nlet_string(self):
        return "abc"

