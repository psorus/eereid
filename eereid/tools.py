import numpy as np
from tqdm import tqdm
import json


def uqdm(iterable,rounde=4,*args,**kwargs):
    t=tqdm(iterable,*args,**kwargs)
    def func(tex,rounde=rounde,t=t):
        text=tex
        if type(text) is float or type(text) is np.float64:
            text=round(text,rounde)
            prekomma,postkomma=str(text).split('.')
            while len(postkomma)<rounde:
                postkomma+='0'
            text=prekomma+'.'+postkomma
        t.set_description_str(str(text))
        return tex
    for zw in t:
        yield zw,func



def datasplit(x,y, mods, novelty=False):
    f_train=mods("train_fraction",0.6)
    f_novel=mods("novel_fraction",0.1)
    f_que=mods("query_fraction",0.5)
    seed=mods("datasplit_seed",42)
    if novelty==False:
        f_novel=0.0
    
    classes=list(set(list(y)))
    n_classes=len(classes)
    rnd = np.random.RandomState(seed)
    rnd.shuffle(classes)

    n_train=int(f_train*n_classes)
    n_novel=int(np.ceil(f_novel*n_classes))
    n_test=n_classes-n_train-n_novel

    train_cls=classes[:n_train]
    test_cls=classes[n_train:n_train+n_test]
    novel_cls=classes[n_train+n_test:]

    class_to_samples={cls:[] for cls in classes}
    for xx,yy in zip(x,y):
        class_to_samples[yy].append(xx)

    class_to_samples={cls:np.array(val) for cls,val in class_to_samples.items()}

    train_x=np.concatenate([class_to_samples[cls] for cls in train_cls])
    train_y=np.concatenate([np.repeat(cls, len(class_to_samples[cls])) for cls in train_cls])

    idx=np.arange(len(train_x))
    rnd.shuffle(idx)
    train_x=train_x[idx]
    train_y=train_y[idx]

    if len(novel_cls)>0:
        novel_x=np.concatenate([class_to_samples[cls] for cls in novel_cls])
        idx=np.arange(len(novel_x))
        rnd.shuffle(idx)
        novel_x=novel_x[idx]


    que_x,que_y,gal_x,gal_y=[],[],[],[]
    for cls in test_cls:
        idx=np.arange(len(class_to_samples[cls]))
        rnd.shuffle(idx)
        que=idx[:int(f_que*len(idx))]
        gal=idx[int(f_que*len(idx)):]
        que_x.append(class_to_samples[cls][que])
        que_y.append(np.repeat(cls, len(que)))
        gal_x.append(class_to_samples[cls][gal])
        gal_y.append(np.repeat(cls, len(gal)))

    que_x=np.concatenate(que_x, axis=0)
    que_y=np.concatenate(que_y, axis=0)
    gal_x=np.concatenate(gal_x, axis=0)
    gal_y=np.concatenate(gal_y, axis=0)

    idx=np.arange(len(gal_x))
    rnd.shuffle(idx)
    gal_x=gal_x[idx]
    gal_y=gal_y[idx]

    idx=np.arange(len(que_x))
    rnd.shuffle(idx)
    que_x=que_x[idx]
    que_y=que_y[idx]

    if len(novel_cls)==0:
        return train_x,train_y,que_x,que_y,gal_x,gal_y
    else:
        return train_x,train_y,que_x,que_y,gal_x,gal_y,novel_x

def crossvalidation(x,y, mods, novelty=False):
    folds=mods("folds",5)
    folds_novel=mods("novel_folds",1)
    folds_train=mods("train_folds",3)
    f_que=mods("query_fraction",0.5)
    seed=mods("datasplit_seed",42)
    if novelty==False:
        folds_novel=0
    folds_test=folds-folds_train-folds_novel

    
    classes=list(set(list(y)))
    n_classes=len(classes)
    rnd = np.random.RandomState(seed)
    rnd.shuffle(classes)

    foldcls=[[] for i in range(folds)]
    for i,cls in enumerate(classes):
        foldcls[i%folds].append(cls)

    for i in range(folds):
        train_folds=[(i+j)%folds for j in range(folds_train)]
        test_folds=[(i+j+folds_train)%folds for j in range(folds_test)]
        novel_folds=[(i+j+folds_train+folds_test)%folds for j in range(folds_novel)]
        train_cls=[cls for fold in train_folds for cls in foldcls[fold]]
        test_cls=[cls for fold in test_folds for cls in foldcls[fold]]
        novel_cls=[cls for fold in novel_folds for cls in foldcls[fold]]

        class_to_samples={cls:[] for cls in classes}
        for xx,yy in zip(x,y):
            class_to_samples[yy].append(xx)
    
        class_to_samples={cls:np.array(val) for cls,val in class_to_samples.items()}
    
        train_x=np.concatenate([class_to_samples[cls] for cls in train_cls])
        train_y=np.concatenate([np.repeat(cls, len(class_to_samples[cls])) for cls in train_cls])

        idx=np.arange(len(train_x))
        rnd.shuffle(idx)
        train_x=train_x[idx]
        train_y=train_y[idx]

        if novelty:
            novel_x=np.concatenate([class_to_samples[cls] for cls in novel_cls])
            idx=np.arange(len(novel_x))
            rnd.shuffle(idx)
            novel_x=novel_x[idx]

        que_x,que_y,gal_x,gal_y=[],[],[],[]
        for cls in test_cls:
            idx=np.arange(len(class_to_samples[cls]))
            rnd.shuffle(idx)
            que=idx[:int(f_que*len(idx))]
            gal=idx[int(f_que*len(idx)):]
            que_x.append(class_to_samples[cls][que])
            que_y.append(np.repeat(cls, len(que)))
            gal_x.append(class_to_samples[cls][gal])
            gal_y.append(np.repeat(cls, len(gal)))

        que_x=np.concatenate(que_x, axis=0)
        que_y=np.concatenate(que_y, axis=0)
        gal_x=np.concatenate(gal_x, axis=0)
        gal_y=np.concatenate(gal_y, axis=0)
    
        idx=np.arange(len(gal_x))
        rnd.shuffle(idx)
        gal_x=gal_x[idx]
        gal_y=gal_y[idx]
    
        idx=np.arange(len(que_x))
        rnd.shuffle(idx)
        que_x=que_x[idx]
        que_y=que_y[idx]

        if novelty:
            yield train_x,train_y,que_x,que_y,gal_x,gal_y,novel_x
        else:
            yield train_x,train_y,que_x,que_y,gal_x,gal_y

def build_triplets(x,y,mods):
    count=mods("triplet_count",10000)

    classes=list(set(list(y)))
    class_to_sample={cls:[] for cls in classes}

    for xx,yy in zip(x,y):
        class_to_sample[yy].append(xx)

    triplets=[]
    for i in range(count):
        cls1,cls2=np.random.choice(classes, 2, replace=False)
        if len(class_to_sample[cls1])<2 or len(class_to_sample[cls2])==0:
            print("failed ",i)
            continue

        a,b=np.random.choice(np.arange(len(class_to_sample[cls1])), 2, replace=False)
        c=np.random.choice(np.arange(len(class_to_sample[cls2])))
        a=class_to_sample[cls1][a]
        b=class_to_sample[cls1][b]
        c=class_to_sample[cls2][c]

        triplets.append((a,b,c))

    return np.array(triplets)

def _choose_one(lis):
    dex=np.random.choice(len(lis))
    return lis[dex]

def _build_Nlets(x,y,generator,mods,count_multiplier=1.0):
    count=mods("Nlet_count",mods("triplet_count",10000))
    count=int(count*count_multiplier)
    generator=[gen for gen in generator]
    N=len(generator)
    types=list(set(generator))
    t=len(types)
    toi={typ:i for i,typ in enumerate(types)}

    classes=list(set(list(y)))
    class_to_sample={cls:[] for cls in classes}

    for xx,yy in zip(x,y):
        class_to_sample[yy].append(xx)

    Nlets=[]
    while len(Nlets)<count:
        clss=np.random.choice(classes,t,replace=False)
        #print("clss",clss)
        #print("toi",toi)
        #print("generator",generator)
        #print("types",types)
        #print("types,clss,toi,gen")
        objs=[_choose_one(class_to_sample[clss[toi[gen]]]) for gen in generator]
        Nlets.append(objs)
    
    return np.array(Nlets)

def build_Nlets(x,y,generator,mods):
    subg=generator.split("/")
    parts=[_build_Nlets(x,y,sub,mods,1/len(subg)) for sub in subg]
    ys=np.concatenate([np.repeat(i,len(part)) for i,part in enumerate(parts)])
    xs=np.concatenate(parts,axis=0)
    ys=ys.astype(np.float32)
    return xs,ys





def rank1(que_emb,que_y,gal_emb,gal_y, distance):
    rank1=0
    for i,func in uqdm(range(len(que_emb))):
        dist=distance.multi_distance(gal_emb,que_emb[i])#np.linalg.norm(val_emb[i]-gal_emb, axis=1)
        idx=np.argsort(dist)
        if gal_y[idx[0]]==que_y[i]:
            rank1+=1
        func(rank1/(i+1))
    return rank1/len(que_emb)

class anyrank():
    def __init__(self,hits,**kwargs):
        self.hits=hits
        self.kwargs=kwargs
    def rankN(self,N):
        return np.mean(self.hits<=N)
    def __call__(self,N):
        if type(N) is str and N in self.kwargs:
            return self.kwargs[N]
        return self.rankN(N)
    def summarize(self):
        dict1 = {f"rank-{i}": self.rankN(i) for i in [1, 2, 3, 5, 10]}
        dict2 = self.kwargs
        merged_dict = dict1.copy()
        merged_dict.update(dict2)
        return merged_dict
    def __repr__(self):
        return json.dumps(self.summarize(),indent=2)
    def __add__(self,other):
        if type(other) is statistics:
            other.add(self)
        else:
            other=statistics(self,other)
        return other
    def __setitem__(self,key,value):
        self.kwargs[key]=value
    def __getitem__(self,key):
        return self(kwargs[key])


class statistics():
    def __init__(self,*res):
        self.res=list(res)

    def add(self,*res):
        for r in res:
            self.res.append(r)

    def eval(self,*args):
        values=[r(*args) for r in self.res]
        return np.mean(values),np.std(values)/np.sqrt(len(values))

    def additional(self,name):
        return np.mean([r(name) for r in self.res]),np.std([r(name) for r in self.res])/np.sqrt(len(self.res))

    def __call__(self,*args):
        if len(args)>0 and type(args[0]) is str:
            return self.additional(args[0])
        return self.eval(*args)

    def __add__(self,other):
        if type(other) is statistics and type(self) is statistics:
            self.add(*other.res)
            return self
        if type(other) is statistics:
            other.add(self)
            return other
        else:
            self.add(other)
            return self

    def inf(self,*args):
        mn,std=self(*args)
        return f"{mn:.4f}+-{std:.4f}"
    def list_additionals(self):
        return [key for key in self.res[0].kwargs.keys()]
    def summarize(self):
        dict1 = {f"rank-{i}": self.inf(i) for i in [1, 2, 3, 5, 10]}
        dict2 = {key: self.inf(key) for key in self.list_additionals()}
        merged_dict = dict1.copy()
        merged_dict.update(dict2)
        return merged_dict
    def __repr__(self):
        return json.dumps(self.summarize(),indent=2)

#def rankN(val_emb,val_y,gal_emb,gal_y,distance, novelty=None):
#    hits=[]
#    rank1=0
#    for i,func in uqdm(range(len(val_emb))):
#        dist=distance.multi_distance(gal_emb,val_emb[i])#np.linalg.norm(val_emb[i]-gal_emb, axis=1)
#        idx=np.argsort(dist)
#        firsthit=int(np.where(gal_y[idx]==val_y[i])[0][0])
#        if firsthit==0:
#            rank1+=1
#        hits.append(firsthit+1)
#        func(rank1/(i+1))
#    return anyrank(np.array(hits))

def compute_ap(ranks, n_relevant):
    """
    Compute the average precision (AP) given the ranks of the relevant items and total number of relevant items.
    Args:
    - ranks: A list of ranks where relevant items appear.
    - n_relevant: Total number of relevant items in the gallery.
    
    Returns:
    - ap: The average precision (AP) score.
    """
    if n_relevant == 0:
        return 0.0

    ranks = np.asarray(ranks) + 1  # Converting to 1-based index
    precisions = np.arange(1, len(ranks) + 1) / ranks
    ap = np.sum(precisions) / n_relevant
    return ap

def rankN(que_emb, que_y, gal_emb, gal_y, distance, novelty=None):
    hits = []
    rank1 = 0
    ap_scores = []

    for i, func in uqdm(range(len(que_emb))):
        dist = distance.multi_distance(gal_emb, que_emb[i])  # np.linalg.norm(val_emb[i] - gal_emb, axis=1)
        idx = np.argsort(dist)
        
        relevant_indices = np.where(gal_y[idx] == que_y[i])[0]
        
        if len(relevant_indices) > 0:
            firsthit = relevant_indices[0]
            if firsthit == 0:
                rank1 += 1
            hits.append(firsthit + 1)
            
            # Compute AP for this query
            ap = compute_ap(relevant_indices, np.sum(gal_y == que_y[i]))
            ap_scores.append(ap)
        else:
            hits.append(len(gal_y) + 1)  # No hit case
        
        func(rank1 / (i + 1))
    
    # Compute mean AP (mAP)
    mAP = np.mean(ap_scores)
    #print(f"Mean Average Precision (mAP): {mAP:.4f}")

    return anyrank(np.array(hits), mAP=mAP)






