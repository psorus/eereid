from eereid.models.wrapmodel import wrapmodel
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

class modelzoo(wrapmodel):
    def __init__(self,model, *args, include_top=False, freeze=False, **kwargs):
        self.zoomodel=model
        self.include_top=include_top
        self.freeze=freeze
        self.args=args
        self.kwargs=kwargs
        super().__init__("modelzoo")

    def build_submodel(self,input_shape, mods):
        add_layers=mods("add_layer_count",1)
        activation=mods("activation","relu")
        nodes=mods("nodes_per_layer",256)
        outputs=mods("output_size",100)
        global_average_pooling=mods("global_average_pooling",False)
        freeze=mods("freeze",self.freeze)

        base_model = self.zoomodel(*self.args, include_top=False, **self.kwargs)

        x = base_model.output
        if global_average_pooling:
            x = GlobalAveragePooling2D()(x)
        
        for i in range(add_layers):
            x = Dense(nodes, activation=activation)(x)
        
        predictions = Dense(outputs, activation='linear')(x)
        
        if freeze:
            for layer in base_model.layers:
                layer.trainable = False

        self.submodel = Model(inputs=base_model.input, outputs=predictions)







