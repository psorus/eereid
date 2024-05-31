from eereid.modifier.modifier import modifier

class crossval(modifier):
    def __init__(self, folds=None):
        self.folds = folds
        super().__init__(label="crossval",value=True)

    def save(self,pth,index):
        super().save(pth,index,folds=self.folds)

    def additional(self):
        if self.folds is None:
            return {}
        else:
            return {"folds":self.folds}




