from eereid.datasets.dataset import dataset
import numpy as np
from tensorflow.keras.datasets import cifar10 as cifartf


class cifar(dataset):
    def __init__(self):
        super().__init__("cifar")

    def load_raw(self):
        (x,y),(tx,ty)=cifartf.load_data()

        x=np.concatenate([x,tx])
        y=np.concatenate([y,ty])

        y=y[:,0]

        return x,y

    def input_shape(self):
        return (32,32,3)
    
    def sample_count(self):
        return 60000

    def explain(self):return "CIFAR-10 data loader"


if __name__=="__main__":
    m=cifar()
    x,y=m.load_data()

    print(x.shape)



