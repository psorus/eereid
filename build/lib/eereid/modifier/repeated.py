from eereid.modifier.modifier import modifier

class repeated(modifier):
    def __init__(self, n=True):
        self.n = n
        super().__init__(label="repeated",value=n)

    def save(self,pth,index):
        super().save(pth,index,n=self.n)





