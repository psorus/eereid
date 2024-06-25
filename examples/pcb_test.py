import eereid as ee
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.applications import ResNet50

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")     #Get the current date and time

g=ee.ghost(model=ee.models.modelzoo(ResNet50),              #Use a simple convolutional model
           dataset=ee.datasets.metal(),         #Train on mnist dataset
           loss=ee.losses.triplet(),   #Use Extended Triplet loss (+d(p,n)) for training
           #prepros=[ee.prepros.resize((32,32)), ee.prepros.subsample(0.1)],   #To speed up training, use only 10% of samples
           triplet_count=1000,                   #To speed up training, use only 100 triplets for training
           crossval=False,
           epochs=100,
           patience=10,
           batch_size=20,
           step_size=20)                       #Enable crossvalidation


g["log_file"]=f"logz_{now}"     #Save training log
g["freeze"]=True
g["pcb"]=True

acc=g.evaluate()        #Evaluate the model

embed=g.plot_embeddings()
plt.show()

loss=g.plot_loss()
plt.show()

print(acc)