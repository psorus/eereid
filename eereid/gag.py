

class gag(object):
    def __init__(self,name):
        self.name=name

    def save(self,pth):
        raise NotImplementedError("Method not implemented")

    def savehelper(self,pth,fn,**kwargs):
        os.makedirs(pth,exist_ok=True)
        with open(os.path.join(pth,fn),'w') as f:
            json.dump({'name':self.name,**kwargs},f,indent=2)   

    def savehelper2(self,pth,ident,index,**kwargs):
        os.makedirs(pth,exist_ok=True)
        spth=os.path.join(pth,ident)
        os.makedirs(spth,exist_ok=True)
        with open(os.path.join(spth,f'{index}.json'),'w') as f:
            json.dump({'name':self.name,"index":index,"type":ident,**kwargs},f,indent=2)

    def species(self):
        raise NotImplementedError("Method not implemented")

    def explain(self):
        return "This is a generic GAG object, specifing some part of a reid model"

    def _add_tags(self, x):
        lines=x.split('\n')
        lines=[f"    {line}" for line in lines]
        return '\n'.join(lines)



