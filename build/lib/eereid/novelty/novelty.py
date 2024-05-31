from eereid.gag import gag

class novelty(gag):
    def __init__(self,name):
        super().__init__(name)
        self.trained=False

    def create_model(self,normal):
        """trains a model to predict novelty"""
        raise NotImplementedError

    def predict(self,samples):
        """predicts novelty of a given input or set of inputs"""
        raise NotImplementedError

    def save(self,pth,**kwargs):
        self.savehelper(pth,"novelty.json",**kwargs)

    def species(self):return "novelty"
