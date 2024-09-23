from eereid.models.model import model
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class extending_layer(keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=0)

class concat_layer(keras.layers.Layer):
    def call(self, inputs):
        return tf.concat(inputs,axis=0)


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
        samples=[extending_layer()(sample) for sample in samples]
        #outp=K.concatenate(samples,axis=0)
        outp=concat_layer()(samples)
        self.model=keras.models.Model(inputs=inp2,outputs=outp)

    def explain(self):
        return "Inheritable generic class creating siamese neural network wrapper"






