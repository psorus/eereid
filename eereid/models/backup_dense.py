from eereid.models.model import model
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K



class simple_dense(model):
    def __init__(self):
        super().__init__("simple_dense")

    def build(self,input_shape, mods):
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

        inp2=keras.layers.Input(shape=[3]+list(input_shape))
        anchor=inp2[0]
        positive=inp2[1]
        negative=inp2[2]
        anchor=self.submodel(anchor)
        positive=self.submodel(positive)
        negative=self.submodel(negative)
        anchor=K.expand_dims(anchor,axis=0)
        positive=K.expand_dims(positive,axis=0)
        negative=K.expand_dims(negative,axis=0)
        outp=K.concatenate([anchor,positive,negative],axis=0)
        self.model=keras.models.Model(inputs=inp2,outputs=outp)






