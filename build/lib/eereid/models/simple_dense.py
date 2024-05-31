from eereid.models.wrapmodel import wrapmodel
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K



class simple_dense(wrapmodel):
    def __init__(self):
        super().__init__("simple_dense")

    def build_submodel(self,input_shape, mods):
        layers=mods("layer_count",3)
        activation=mods("activation","relu")
        nodes=mods("nodes_per_layer",256)
        outputs=mods("output_size",100)

        inp=keras.layers.Input(shape=input_shape)
        q=keras.layers.Flatten()(inp)
        for i in range(layers):
            q=keras.layers.Dense(nodes,activation=activation)(q)
        q=keras.layers.Dense(outputs,activation="linear")(q)

        self.submodel=keras.models.Model(inputs=inp,outputs=q)






