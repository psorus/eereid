import eereid as ee
import matplotlib.pyplot as plt
import datetime
import numpy as np

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")     #Get the current date and time

def func(x):
    x=np.array(x)
    x=np.abs(x)
    while len(x.shape)>1:
        x=np.std(x,axis=-1)
    return x

def func1(x):
    return np.mean(x,axis=(1,2))
def func2(x):
    return np.std(x,axis=(1,2))

                      #Enable crossvalidation
g=ee.ghost(ee.models.graph(),ee.prepros.grapho(func=func))
c=ee.ghost()
m=ee.ghost(ee.prepros.apply_func(func1))
s=ee.ghost(ee.prepros.apply_func(func2))

models=[g,c,m,s]


for model in models:
    model["triplet_count"]=1000
    model(ee.prepros.subsample(0.001))
    dataset=ee.datasets.metal()
    batch_size=100
    step_size=100
    epochs=100
    patience=10
    crossval=False


g=ee.haunting(*models)

g["log_file"]=f"logz_{now}"     #Save training log

acc=g.evaluate()        #Evaluate the model

#embed=g.plot_embeddings()
#plt.show()

#loss=g.plot_loss()
#plt.show()

print(acc)