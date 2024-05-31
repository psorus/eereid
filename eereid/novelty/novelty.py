from gag import gag

class novelty(gag):
    def __init__(self):
        super().__init__()
        self.trained=False

    def create_model(self,normal):
        """trains a model to predict novelty"""
        raise NotImplementedError

    def predict(self,normal):
        """predicts novelty of a given input or set of inputs"""
        raise NotImplementedError

    def save(self,pth,**kwargs):
        self.savehelper(pth,"novelty.json",**kwargs)

    def species(self):return "novelty"
