from eereid.gag import gag

class model(gag):
    def __init__(self,name):
        super().__init__(name)
        self.model=None
        self.submodel=None
        self.trained=False

    def build(self,input_shape, siamese_count, mods):
        raise NotImplementedError

    def save(self,pth,**kwargs):
        self.savehelper(pth,"model.json",**kwargs)

    def species(self):return "model"

    def compile(self,loss,*args,**kwargs):
        return self.model.compile(*args,loss=loss,**kwargs)

    def summary(self):
        print("submodel:")
        self.submodel.summary()
        print("model:")
        self.model.summary()

    def fit(self,triplets,*args,**kwargs):
        ret= self.model.fit(triplets,triplets,*args,**kwargs)
        self.trained=True
        return ret

    def embed(self,data):
        return self.submodel.predict(data)

    def save_model(self,pth):
        self.submodel.save(pth)
