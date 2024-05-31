from eereid.datasets.dataset import dataset
import numpy as np
from tensorflow.keras.datasets import fashion_mnist as fmnisttf


class fmnist(dataset):
    def __init__(self):
        super().__init__("fmnist")

    def load_raw(self):
        (x,y),(tx,ty)=fmnisttf.load_data()

        x=np.concatenate([x,tx])
        y=np.concatenate([y,ty])

        return x,y

    def input_shape(self):
        return (28,28)
    
    def sample_count(self):
        return 70000


if __name__=="__main__":
    m=fmnist()
    x,y=m.load_data()

    print(x.shape)



