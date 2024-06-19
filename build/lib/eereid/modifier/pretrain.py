from eereid.modifier.modifier import modifier

class pretrain(modifier):
    def __init__(self,**kwargs):
        self.kwargs = dict(kwargs)
        super().__init__(label="pretrain",value=n)

    def save(self,pth,index):
        super().save(pth,index,**self.kwargs)

    def additional(self):
        return self.kwargs

    def explain(self):
        return "Modifier enabling pretraining with the following parameters: {}".format(self.kwargs)





