from eereid.models.model import model
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

try:
    from spektral.layers import GCNConv
    #from spektral.layers import AsymCheegerCutPool as Pool
    #from spektral.layers import DiffPool as Pool
except ImportError:
    from eereid.importhelper import importhelper
    GCNConv=importhelper("spektral", "simple_graph layer")


class advanced_graph(model):
    def __init__(self):
        super().__init__("advanced_graph")

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
        graph_activation=mods("graph_activation", "same")
        if graph_activation=="same":
            graph_activation=activation
        filters = mods("filters", 64)
        outputs = mods("output_size", 100)
        hidden_dense=mods("hidden_dense", 3)
        hidden_dense_size=mods("hidden_dense_size", 256)
        gcl=mods("graph_layer", GCNConv)
        conv_per_block=mods("graph_conv_per_block", 1)
        #pool=mods("graph_pool", Pool)
        #poolf=mods("pooling_factor", 0.5)
        pooler=mods("graph_pooling",K.mean)

        #input shape should be (nodes, nodes+features)
        nodes = input_shape[0]
        features = input_shape[1]-nodes
    
        # Define input layers for node features and adjacency matrix
        adjacency_matrix = keras.layers.Input(shape=(nodes, nodes))
        node_features = keras.layers.Input(shape=(nodes, features))
    
        q = node_features
    
        for i in range(layers_count):
            for j in range(conv_per_block):
                q = gcl(filters, activation=graph_activation)([q, adjacency_matrix])
            #if pool is not None:
            #    k=int(np.ceil(nodes*poolf))
            #    q, adjacency_matrix,*_ = pool(k)([q, adjacency_matrix])
            #    nodes=k
    
        if pooler is None:
            q = keras.layers.Flatten()(q)
        else:
            q=pooler(q,axis=1)
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



