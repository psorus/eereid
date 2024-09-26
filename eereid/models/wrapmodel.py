from eereid.models.model import model
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K



class wrapmodel(model):
    def __init__(self,name):
        super().__init__("wrap_"+name)

    def build_submodel(self,input_shape,mods):
        raise NotImplementedError

    def build(self,input_shape, siamese_count, mods):
        self.build_submodel(input_shape,mods)

        inp2=keras.layers.Input(shape=[siamese_count]+list(input_shape))
        
        samples=[inp2[:,i] for i in range(siamese_count)]
        samples=[self.submodel(sample) for sample in samples]        
        samples = [keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=0))(sample) for sample in samples]
        outp = keras.layers.Concatenate(axis=0)(samples)
        
        self.model=keras.models.Model(inputs=inp2,outputs=outp)

    def explain(self):
        return "Inheritable generic class creating siamese neural network wrapper"
