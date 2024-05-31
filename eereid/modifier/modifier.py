from eereid.gag import gag

class modifier(gag):
    def __init__(self,label,value):
        super().__init__("modifier_"+label)
        self.label = label
        self.value = value

    def save(self,pth,index,**kwargs):
        self.savehelper2(pth,"modifier",index,label=self.label,value=self.value,**kwargs)

    def species(self):return "modifier"

    def ident(self):
        return self.label

    def additional(self):
        return {}
