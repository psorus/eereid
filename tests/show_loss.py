import eereid as ee
from plt import *

g=ee.ghost(model=ee.models.conv(),              #Use a simple convolutional model
           dataset=ee.datasets.mnist(),         #Train on mnist dataset
           loss=ee.losses.extended_triplet(),   #Use Extended Triplet loss (+d(p,n)) for training
           novelty=ee.novelty.distance(),       #Evaluate also novelty detection. Far away=novel
           preproc=ee.prepros.subsample(0.1),   #To speed up training, use only 10% of samples
           triplet_count=100,                   #To speed up training, use only 100 triplets for training
           subsample=0.1,
           crossval=False)                       #Enable crossvalidation

acc=g.evaluate()                                #Evaluate the model

print(acc)

plt.figure(figsize=(10,5))

g.plot_loss()                                  #Plot the loss

plt.savefig("last.png")

plt.show()


