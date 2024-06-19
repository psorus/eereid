import time

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


    def log(self,message,importance=1):
        #the higher the importance, the more important the message is. If importance over border importance, the message will be printed
        log_file=self("log_file","")
        print_border=self("log_level_print",2)
        log_border=self("log_level_log",0)
        if importance>=print_border:
            print(message)
        if not (log_file=="") and importance>=log_border:
            with open(log_file,"a") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S")+": "+message+"\n")
                #f.write(message+"\n")










