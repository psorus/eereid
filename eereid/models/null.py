from eereid.gag import gag
from eereid.models.model import model

class null(model):
    def __init__(self):
        super().__init__("null")
        self.model=None
        self.submodel=None
        self.trained=False

    def build(self,input_shape, siamese_count, mods):
        pass

    def save(self,pth,**kwargs):
        raise NotImplementedError("No model is not savable")

    def species(self):return "model"

    def compile(self,loss,*args,**kwargs):
        pass

    def summary(self):
        print("Null model")

    def fit(self,triplets,labels,*args,**kwargs):
        pass

    def embed(self,data):
        return data

    def save_model(self,pth):
        raise NotImplementedError("No model is not savable")

    def explain(self):
        return "Null model just copying data"
