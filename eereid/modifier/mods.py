
class mods(object):

    def __init__(self, *modifier):
        self.dic={}

        self.add(*modifier)

    def __call__(self, name, usual=None):
        if name in self.dic:
            ret= self.dic[name]
            if hasattr(ret,"value"):
                return ret.value
            return ret
        elif not usual is None:
            return usual
        else:
            raise AttributeError

    def hasattr(self, name):
        return name in self.dic and bool(self.dic[name])

    def add(self, *modifier):
        for mod in modifier:
            self.dic[mod.ident()]=mod

    def call(self,*modifier):
        self.add(*modifier)

    def set_key(self, key, value):
        self.dic[key]=value
    def __setitem__(self, key, value):
        self.dic[key]=value

    def explain(self):
        def subexplain(q):
            if hasattr(q,"explain"):
                return q.explain()
            return str(q)

        return "Modifier:"+"\n".join([f"    {k}: {subexplain(v)}" for k,v in self.dic.items()])

