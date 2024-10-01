from eereid.models.wrapmodel import wrapmodel
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, AveragePooling2D, Conv2D

class modelzoo(wrapmodel):
    def __init__(self,model, *args, include_top=False, freeze=False, **kwargs):
        self.zoomodel=model
        self.include_top=include_top
        self.freeze=freeze
        self.args=args
        self.kwargs=kwargs
        super().__init__("modelzoo")

    def build_submodel(self,input_shape, mods):
        add_layers=mods("add_layer_count",4)
        activation=mods("activation","relu")
        nodes=mods("nodes_per_layer",256)
        outputs=mods("output_size",100)
        global_average_pooling=mods("global_average_pooling", False)
        pcb=mods("pcb", False)
        freeze=mods("freeze",self.freeze)
        input_layer = tf.keras.layers.Input(shape=input_shape)
        base_model = self.zoomodel(*self.args, include_top=False, **self.kwargs, input_tensor=input_layer)

        x = base_model.output
        y = base_model.output_shape[1:3]
        print("shape pre global_", y)
        print("shape pre global", x.shape)
        if global_average_pooling:
            x = AveragePooling2D(y)(x)
            print("shape after global pooling", x.shape)
            x = Conv2D(256, 1)(x)
            print("shape after conv2d", x.shape)

        if pcb:
            # [B, H, W, C]
            shape = base_model.output_shape
            stripe_h = int(shape[1] / add_layers)

            local_feat_list = []
            logits_list = []
            #print(stripe_h, x.shape)
            for i in range(add_layers):
                temp = x[:,i * stripe_h : (i+1)*stripe_h, :, :]
                #print(temp.shape)
                local_feat = tf.nn.avg_pool2d(temp, stripe_h, shape[2], 'VALID')
                #print("local feat", local_feat.shape)
                local_feat = Conv2D(256, 1)(local_feat)
                local_feat = Flatten()(local_feat)
                local_feat_list.append(local_feat)

                logits = Dense(nodes,activation='softmax')(local_feat)
                #print("logits", logits.shape)
                logits_list.append(logits)

            predictions = tf.concat(logits_list, axis=-1)
            #print("pred", predictions.shape)

        if not pcb:
            x=Flatten()(x)
            print(x.shape)
            predictions = Dense(outputs, activation='linear')(x)
        # predictions = Dense(outputs, activation='softmax')(x) #softmax instead of linear
        
        if freeze:
            for layer in base_model.layers:
                layer.trainable = False

        self.submodel = Model(inputs=base_model.input, outputs=predictions)

    def explain(self):
        return f"Modelzoo loader gag, using the base model {self.zoomodel.__name__}." + ("Freezing the pretrained weights." if self.freeze else "")
