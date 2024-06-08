from eereid.datasets.dataset import dataset
import numpy as np
from tensorflow.keras.datasets import mnist as mnisttf


class mnist(dataset):
    def __init__(self):
        super(mnist, self).__init__("mnist")

    def load_raw(self):
        (x,y),(tx,ty)=mnisttf.load_data()

        x=np.concatenate([x,tx])
        y=np.concatenate([y,ty])

        return x,y

    def input_shape(self):
        return (28,28)
    
    def sample_count(self):
        return 70000

    def explain(self):
        return "MNIST data loader"


if __name__=="__main__":
    m=mnist()
    x,y=m.load_data()

    print(x.shape)



