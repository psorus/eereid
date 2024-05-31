from eereid.models.wrapmodel import wrapmodel
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K



class custom_model(wrapmodel):
    def __init__(self,model):
        super().__init__("custom_model")
        self.submodel=model

    def build_submodel(self,input_shape, mods):
        #assert (self.submodel.input_shape==input_shape), f"The provided model has a different input shape than the data ({self.submodel.input_shape} vs {input_shape})"
        pass






