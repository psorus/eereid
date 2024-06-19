from eereid.losses.loss import loss

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

class custom_loss(loss):
    def __init__(self, func,Nlet="aab"):
        super().__init__("custom_loss")
        self.func=func
        self.Nlet=Nlet

    def build(self,mods):
        return self.func

    def save(self,pth):
        raise Exception("Not implement(ed/able)")
        super().save(pth,Nlet=self.Nlet)
        
    def Nlet_string(self):
        return self.Nlet

    def explain(self):
        return "Custom loss function."
