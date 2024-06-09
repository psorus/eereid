import numpy as np
import eereid as ee

from tensorflow.keras.applications import MobileNetV2

from plt import *


g=ee.ghost()



#just to speed up the evaluation
g(ee.prepros.subsample(0.1))
g["triplet_count"]=100


acc=g.evaluate()
print(acc)

plt.figure(figsize=(10,6))

g.plot_embeddings()

plt.show()


