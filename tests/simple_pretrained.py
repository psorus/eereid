import numpy as np
import eereid as ee

from tensorflow.keras.applications import MobileNetV2


g=ee.ghost(model=ee.models.modelzoo(MobileNetV2, weights="imagenet"),
           dataset=ee.datasets.mnist(),
           prepros=[ee.prepros.resize((32,32)),ee.prepros.add_color()])

#lets freeze the pretrained layers
g["freeze"]=True


#just to speed up the evaluation
g(ee.prepros.subsample(0.1))
g["triplet_count"]=1000


acc=g.evaluate()
print(acc)

