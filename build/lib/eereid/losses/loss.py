from eereid.gag import gag

class loss(gag):
    def __init__(self,name):
        super().__init__(name)

    def build(self, mods):
        raise NotImplementedError

    def save(self,pth,**kwargs):
        self.savehelper(pth,"loss.json",**kwargs)

    def species(self):return "loss"

    def Nlet_string(self):
        raise NotImplementedError

    def siamese_count(self):
        return len(self.Nlet_string())
