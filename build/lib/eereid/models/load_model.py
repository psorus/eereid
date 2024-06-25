from eereid.models.wrapmodel import wrapmodel
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K



class load_model(wrapmodel):
    def __init__(self,pth):
        super().__init__("load_model")
        self.pth=pth

    def build_submodel(self,input_shape, mods):
        self.submodel=keras.models.load_model(self.pth)
        self.trained=True

    def fit(self,triplets,*args,**kwargs):
        self.trained=True

    def explain(self):
        return f"Model loaded from {self.pth}"




