import eereid as ee
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.applications import ResNet50

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")     #Get the current date and time

g=ee.ghost(model=ee.models.modelzoo(ResNet50),              #Use a simple convolutional model
           dataset=ee.datasets.pallet502(),         #Train on mnist dataset
           loss=ee.losses.triplet(),   #Use Extended Triplet loss (+d(p,n)) for training
           #prepros=[ee.prepros.resize((32,32)), ee.prepros.subsample(0.1)],   #To speed up training, use only 10% of samples
           triplet_count=10000,                   #To speed up training, use only 100 triplets for training
           crossval=False,
           epochs=100,
           patience=10,
           batch_size=20,
           step_size=20)                       #Enable crossvalidation


g["log_file"]=f"logz_{now}"     #Save training log

#g["freeze"]=True
#g["global_average_pooling"]=True
g["pcb"]=True

#g.load_data("502.npz")

acc=g.evaluate()        #Evaluate the model

#g.save_data("502")

embed=g.plot_embeddings()
plt.show()

loss=g.plot_loss()
plt.show()

print(acc)
