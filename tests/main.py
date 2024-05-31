from plt import *
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from eereid.ghost import ghost

from eereid.datasets.mnist import mnist
from eereid.datasets.fmnist import fmnist
from eereid.datasets.cifar import cifar
from eereid.datasets.from_folder import from_folder
from eereid.datasets.market1501 import market1501

from eereid.distances.lN import lN
from eereid.distances.cosine_similarity import cosine_similarity
from eereid.distances.mahalanobis import mahalanobis
from eereid.distances.chisquare import chisquare
from eereid.distances.custom_distance import custom_distance

from eereid.losses.triplet import triplet
from eereid.losses.quadruplet import quadruplet
from eereid.losses.ringloss import ringloss
from eereid.losses.contrastive import contrastive
from eereid.losses.extended_triplet import extended_triplet
from eereid.losses.custom_loss import custom_loss

from eereid.models.simple_dense import simple_dense
from eereid.models.simple_conv import simple_conv

from eereid.prepros.resize import resize
from eereid.prepros.blackwhite import blackwhite
from eereid.prepros.rotations import rotations
from eereid.prepros.flips import flips   
from eereid.prepros.subsample import subsample

from eereid.modifier.crossval import crossval
from eereid.modifier.repeated import repeated

from eereid.ghost import ensemble

g1=ghost(mnist(),lN(2),extended_triplet(),simple_conv())
g2=ghost(lN(2),extended_triplet(),simple_conv())
g1["triplet_count"]=100#0
g2["triplet_count"]=100#0

g=g2+g1
g(subsample(0.1))

#label=lambda fn: int(fn.split("/")[-1].split("_")[0])
#include=lambda fn: fn.endswith(".jpg") and not fn.startswith("-1")
#dataset=from_folder(["/home/psorus/d/test/data_eereid/Market-1501-v15.09.15/bounding_box_train/*.jpg",
#                    "/home/psorus/d/test/data_eereid/Market-1501-v15.09.15/bounding_box_test/*.jpg"],
#                    label,include)
#g(dataset)

#g(market1501())   

#g=g(resize((16,16)))
#g=g(cifar())
#g=g(blackwhite())
#g=g(rotations())
#g=g(flips())
#g=g(fmnist())
#g=g(fmnist()+mnist())
#g=g(mahalanobis())
#g=g(chisquare())
#g=g(custom_distance(lambda x,y:np.max(np.abs(x-y))))
#g=g(quadruplet())
#g=g(ringloss())

#def loss(y_true, y_pred):
#    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
#    positive_dist = K.sum(K.square(anchor - positive), axis=-1)
#    negative_dist = K.sum(K.square(anchor - negative), axis=-1)
#    return K.sum(K.maximum(positive_dist - negative_dist,0), axis=-1)
#g(custom_loss(loss,"bbb"))

#g(contrastive())

#g(crossval())
#g(repeated(2))

g["folds"]=2
g["train_folds"]=1

acc=g.evaluate()
#acc=g.repeated_eval(3)
#acc=g.crossval_eval()
print(acc)

#g.plot_embeddings()



#plt.show()
