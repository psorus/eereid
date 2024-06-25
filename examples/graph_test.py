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

g=ee.ghost(model=ee.models.simple_graph(),              #Use a simple convolutional model
           dataset=ee.datasets.metal(),         #Train on mnist dataset
           loss=ee.losses.extended_triplet(),   #Use Extended Triplet loss (+d(p,n)) for training
           prepros=[ee.prepros.grapho(func=func)],   #To speed up training, use only 10% of samples
           triplet_count=1000,                   #To speed up training, use only 100 triplets for training
           crossval=False,
           batch_size=100,
           step_size=100,
           epochs=100,
           patience=10)                       #Enable crossvalidation


g["log_file"]=f"logz_{now}"     #Save training log

acc=g.evaluate()        #Evaluate the model

embed=g.plot_embeddings()
plt.show()

loss=g.plot_loss()
plt.show()

print(acc)