import eereid as ee
import numpy as np


def func1(x):
    return np.mean(x,axis=(1,2))
def func2(x):
    return np.std(x,axis=(1,2))


g=ee.ghost(ee.models.graph(),ee.prepros.grapho())
c=ee.ghost()
m=ee.ghost(ee.prepros.apply_func(func1))
s=ee.ghost(ee.prepros.apply_func(func2))

models=[g,c,m,s]


for model in models:
    model["triplet_count"]=1000
    model(ee.prepros.subsample(0.001))

g=ee.haunting(*models)

acc=g.evaluate()

print(acc)


