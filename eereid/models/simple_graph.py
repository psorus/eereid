from eereid.models.model import model
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

try:
    from spektral.layers import GCNConv
except ImportError:
    from eereid.importhelper import importhelper
    GCNConv=importhelper("spektral", "simple_graph layer")


class simple_graph(model):
    def __init__(self):
        super().__init__("simple_graph")

    def fit(self, triplets, labels, *args,**kwargs):
        #triplets should have shape
        #(-1, 3, nodes, nodes+features)
        nodes=triplets.shape[2]
        Adj=triplets[:,:,0:nodes,0:nodes]
        Features=triplets[:,:,0:nodes,nodes:]
        return super().fit((Adj,Features),labels,*args,**kwargs)

    def embed(self, data):
        #data should have shape
        #(-1, nodes, nodes+features)
        nodes=data.shape[1]
        Adj=data[:,0:nodes,0:nodes]
        Features=data[:,0:nodes,nodes:]
        return super().embed((Adj,Features))



    def build_submodel(self, input_shape, mods):
        layers_count = mods("layer_count", 3)
        activation = mods("activation", "relu")
        filters = mods("filters", 64)
        outputs = mods("output_size", 100)
        hidden_dense=mods("hidden_dense", 3)
        hidden_dense_size=mods("hidden_dense_size", 256)

        #input shape should be (nodes, nodes+features)
        nodes = input_shape[0]
        features = input_shape[1]-nodes
    
        # Define input layers for node features and adjacency matrix
        adjacency_matrix = keras.layers.Input(shape=(nodes, nodes))
        node_features = keras.layers.Input(shape=(nodes, features))
    
        q = node_features
    
        for i in range(layers_count):
            q = GCNConv(filters, activation=activation)([q, adjacency_matrix])
    
        q = keras.layers.Flatten()(q)
        for i in range(hidden_dense):
            q=keras.layers.Dense(hidden_dense_size, activation=activation)(q)
        q = keras.layers.Dense(outputs, activation="linear")(q)

        self.submodel = keras.Model(inputs=[adjacency_matrix,node_features], outputs=q)

    def build(self,input_shape, siamese_count, mods):
        #input shape should be (nodes, nodes+features)
        nodes = input_shape[0]
        features = input_shape[1]-nodes

        self.build_submodel(input_shape,mods)

        inpA=keras.layers.Input(shape=[siamese_count]+[nodes,nodes])
        inpX=keras.layers.Input(shape=[siamese_count]+[nodes,features])

        samplesA=[inpA[:,i] for i in range(siamese_count)]
        samplesX=[inpX[:,i] for i in range(siamese_count)]

        samples=[self.submodel((sA,sX)) for sA,sX in zip(samplesA,samplesX)]
        samples=[K.expand_dims(sample,axis=0) for sample in samples]
        outp=K.concatenate(samples,axis=0)
        self.model=keras.models.Model(inputs=(inpA,inpX),outputs=outp)

    def explain(self):
        return "Creating a simple graph neural network model."



