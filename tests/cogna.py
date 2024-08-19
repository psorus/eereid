import eereid as ee

from spektral.layers import GCNConv
import numpy as np

from tensorflow.keras import backend as K

g=ee.ghost()

g(ee.datasets.mnist())
g(ee.models.advanced_graph())
g(ee.prepros.grapho(consider_count=100))
g(ee.prepros.subsample(0.001))

g["layer_count"]=3
g["activation"]="relu"
g["filters"]=64
g["output_size"]=100
g["hidden_dense"]=3
g["hidden_dense_size"]=100


g["graph_activation"]="same"

g["graph_pooling"]=K.mean #None, K.mean, K.sum, K.std, K.max, K.min


g["graph_layer"]=GCNConv  #https://graphneural.network/layers/convolution/



acc=g.evaluate()

print(acc)


