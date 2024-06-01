import numpy as np
import matplotlib.pyplot as plt

from eereid.tools import datasplit, build_triplets, build_Nlets, rankN, crossvalidation

import tensorflow as tf
from tensorflow import keras

from eereid.modifier.mods import mods

from sklearn.metrics import roc_auc_score

import eereid as ee

class ghost():
    def __init__(self,*tags,dataset=None, distance=None, loss=None, model=None, novelty=None, experiments=None,modifier=None,prepros=None,preprocessing=None, **kwargs):
        #add kwargs 
        self.dataset=None
        self.distance=None
        self.experiments={}
        self.loss=None
        self.model=None
        self.modifier=mods()
        self.novelty=None
        self.prepro={}
        for tag in tags:
            self.add(tag)

        if dataset is not None:
            self.set_dataset(dataset)
        if distance is not None:
            self.set_distance(distance)
        if loss is not None:
            self.set_loss(loss)
        if model is not None:
            self.set_model(model)
        if novelty is not None:
            self.set_novelty(novelty)
        if experiments is not None:
            if type(experiments) is list:
                for experiment in experiments:
                    self.add_experiment(experiment)
            else:
                self.add_experiment(experiments)
        if modifier is not None:
            if type(modifier) is list:
                for mod in modifier:
                    self.add_modifier(mod)
            else:
                self.add_modifier(modifier)
        if prepros is not None:
            if type(prepros) is list:
                for prepro in prepros:
                    self.add_prepro(prepro)
            else:
                self.add_prepro(prepros)
        if preprocessing is not None:
            if type(preprocessing) is list:
                for prepro in preprocessing:
                    self.add_prepro(prepro)
            else:
                self.add_prepro(preprocessing)

        self.add(kwargs)


    def add(self,tag):
        if type(tag) is dict:
            for key,value in tag.items():
                self.modifier[key]=value
            return self
        species=tag.species()
        if species=='dataset':
            self.set_dataset(tag)
        elif species=='distance':
            self.set_distance(tag)
        elif species=='experiment':
            self.add_experiment(tag)
        elif species=='loss':
            self.set_loss(tag)
        elif species=='model':
            self.set_model(tag)
        elif species=='modifier':
            self.add_modifier(tag)
        elif species=='novelty':
            self.set_novelty(tag)
        elif species=='prepro':
            self.add_prepro(tag)
        else:
            raise ValueError("Unknown species encountered",species,tag)
        return self

    def set_dataset(self,dataset):
        self.dataset=dataset
        return self

    def set_distance(self,distance):
        self.distance=distance
        return self

    def set_loss(self,loss):
        self.loss=loss
        return self

    def set_model(self,model):
        self.model=model
        return self

    def set_novelty(self,novelty):
        self.novelty=novelty
        return self

    def add_experiment(self,experiment):
        self.experiments[experiment.ident()]=experiment
        return self

    def add_modifier(self,modifier):
        self.modifier.add(modifier)
        return self

    def add_prepro(self,prepro):
        self.prepro[prepro.ident()]=prepro
        return self


    def __setitem__(self,key,value):
        self.add({key:value})
        return self

    def __call__(self,*tags):
        for tag in tags:
            self.add(tag)
        return self

    def mods(self):
        return self.modifier

    def _preprocess(self):
        allowed=["subsample","resize","blackwhite","add_color"]
        for prepro in allowed:
            if prepro in self.prepro:
                self.x,self.y=self.prepro[prepro].apply(self.x,self.y,self)
    def _preprocess_train(self):
        allowed=["rotations","flips"]
        for prepro in allowed:
            if prepro in self.prepro:
                self.tx,self.ty=self.prepro[prepro].apply(self.tx,self.ty,self)

    def _basic_data_loading(self):
        self.x,self.y=self.dataset.load_data(self.mods())
        self.input_shape=list(self.dataset.input_shape())
        self._preprocess()
        if self.novelty is None:
            self.tx,self.ty,self.vx,self.vy,self.gx,self.gy=datasplit(self.x,self.y,self.mods(),novelty=False)
        else:
            self.tx,self.ty,self.vx,self.vy,self.gx,self.gy,self.nx=datasplit(self.x,self.y,self.mods(),novelty=True)
        self._preprocess_train()
    def _crossval_data_loading(self):
        self.x,self.y=self.dataset.load_data(self.mods())
        self.input_shape=list(self.dataset.input_shape())
        self._preprocess()
        fold=0
        if self.novelty is None:
            for self.tx,self.ty,self.vx,self.vy,self.gx,self.gy in crossvalidation(self.x,self.y,self.mods(),novelty=False):
                self._preprocess_train()
                yield fold
                fold+=1
        else:
            for self.tx,self.ty,self.vx,self.vy,self.gx,self.gy,self.nx in crossvalidation(self.x,self.y,self.mods(),novelty=True):
                self._preprocess_train()
                yield fold
                fold+=1
        
    def _create_model(self):
        self.model.build(self.input_shape,self.loss.siamese_count(),self.mods())
    def _pretrain_prediction(self):
        pretrain_epochs=self.mods()("pretrain_epochs",1)
        pretrain_loss=self.mods()("pretrain_los","categorical_crossentropy")
        pretrain_optimizer=self.mods()("pretrain_optimizer","adam")

        inp=keras.Input(shape=self.input_shape)
        classes=list(set(self.ty))
        classes.sort()
        classes=np.array(classes)
        onehot=np.eye(len(classes))
        class_to_vec={cls:vec for cls,vec in zip(classes,onehot)}
        onehot=np.array([class_to_vec[cls] for cls in self.ty])
        q=self.model.submodel(inp)
        q=keras.layers.Dense(len(classes),activation='softmax')(q)
        model=keras.Model(inp,q)

        model.compile(loss=pretrain_loss,optimizer=pretrain_optimizer)
        model.summary()
        model.fit(self.tx,onehot,epochs=pretrain_epochs)


    def _train_model(self):
        optimizer=self.mods()("optimizer","adam")
        epochs=self.mods()("epochs",10)
        batch_size=self.mods()("batch_size",32)
        #meeeehr
        #lr early stopping val split....

        loss=self.loss.build(self.mods())
        self.model.compile(loss=loss,optimizer=optimizer)

        Nlets=build_Nlets(self.tx,self.ty,self.loss.Nlet_string(),self.mods())

        self.model.fit(Nlets,epochs=epochs,batch_size=batch_size)

    def _embed(self,data):
        #embedding but no preprocess
        return self.model.embed(data)

    def _create_embeddings(self):
        self.emb =self._embed(self.tx)
        self.vemb=self._embed(self.vx)
        self.gemb=self._embed(self.gx)
        if self.novelty is not None:
            self.nemb=self._embed(self.nx)

    def _all_available_data(self):
        ret={}
        poss=["x","y","tx","ty","vx","vy","gx","gy","nx","emb","vemb","gemb","nemb","input_shape"]
        for pos in poss:
            if hasattr(self,pos):
                ret[pos]=getattr(self,pos)
        return ret

    def load_model(self,pth):
        self.set_model(ee.models.load_model(pth))

    def save_data(self,pth):
        np.savez_compressed(pth,**self._all_available_data())

    def load_data(self,pth):
        self.set_dataset(ee.datasets.load_data(pth))

    def _basic_accuracy(self):
        self._create_embeddings()

        distance=self.distance

        acc=rankN(self.vemb,self.vy,self.gemb,self.gy,distance=distance)

        if self.novelty is not None:
            self.novelty.create_model(self.gemb)
            normal=self.vemb
            abnormal=self.nemb
            test=np.concatenate([normal,abnormal])
            label=np.concatenate([np.zeros(len(normal)),np.ones(len(abnormal))])
            acc["auc"]=roc_auc_score(label,self.novelty.predict(test))

        

        return acc

    def evaluate(self):
        self._basic_data_loading()
        if self.mods().hasattr("crossval"):
            return self._crossval_eval()
        elif self.mods().hasattr("repeated"):
            return self._repeated_eval(self.mods()("repeated"))
        else:
            return self._singular_eval()

    def _singular_eval(self):
        self._create_model()
        if self.mods().hasattr("pretrain"):self._pretrain_prediction()
        self._train_model()
        acc=self._basic_accuracy()

        return acc

    def _repeated_eval(self,n):
        if type(n) is bool:
            n=10
        #should be done through modifier
        accs=None
        for i in range(n):
            acc=self._singular_eval()
            if accs is None:
                accs=acc
            else:
                accs=accs+acc
        return accs

    def _crossval_eval(self):
        #should be done through modifier
        accs=None
        for i in self._crossval_data_loading():
            if self.mods().hasattr("repeated"):
                acc=self._repeated_eval(self.mods()("repeated"))
            else:
                acc=self._singular_eval()
            if accs is None:
                accs=acc
            else:
                accs=accs+acc
        return accs


    def plot_embeddings(self):
        from sklearn.decomposition import PCA

        pca=PCA(n_components=2)
        emb=pca.fit_transform(self.emb)
        vemb=pca.transform(self.vemb)
        gemb=pca.transform(self.gemb)

        plt.subplot(1,2,1)

        plt.scatter(emb[:,0],emb[:,1],c=self.ty,alpha=0.5)
        plt.colorbar()

        plt.subplot(1,2,2)
        plt.scatter(vemb[:,0],vemb[:,1],c=self.vy,alpha=0.5)
        plt.scatter(gemb[:,0],gemb[:,1],c=self.gy,alpha=0.5)

        plt.colorbar()

    def __add__(self,other):
        if type(self) is ensemble and type(other) is ensemble:
            return ensemble(*self.objs,*other.objs)
        if type(self) is ensemble and type(other) is not ensemble:
            return self.add_objs(other)
        if type(self) is not ensemble and type(other) is ensemble:
            return other.add_objs(self)
        return ensemble(self,other)

    def save_model(self,path):
        self.model.save_model(path)


class ensemble(ghost):
    def __init__(self,*objs):
        self.objs=list(objs)
        super().__init__()
        self.dataset=self.search_through_objs("dataset")#objs[0].dataset
        self.distance=self.search_through_objs("distance")#objs[0].distance

    def add_objs(self,*objs):
        for obj in objs:
            self.objs.append(obj)
        return self

    def search_through_objs(self,attr):
        for obj in self.objs:
            if hasattr(obj,attr) and getattr(obj,attr) is not None:
                return getattr(obj,attr)
        return None

    def _basic_data_loading(self):
        self.x,self.y=self.dataset.load_data(self.mods())
        self.input_shape=list(self.dataset.input_shape())
        self._preprocess()
        for obj in self.objs:
            obj.x=self.x
            obj.y=self.y
            obj.input_shape=self.input_shape
            obj._preprocess()
            obj.tx,obj.ty,obj.vx,obj.vy,obj.gx,obj.gy=datasplit(obj.x,obj.y,obj.mods())
            obj._preprocess_train()
    def _embed(self,data):
        return np.concatenate([obj.model._embed(data) for obj in self.objs],axis=1)

    def _singular_eval(self):
        for obj in self.objs:
            obj._create_model()
            if obj.mods().hasattr("pretrain"):obj._pretrain_prediction()
            obj._train_model()
        self.vy=self.objs[0].vy
        self.gy=self.objs[0].gy
        acc=self._basic_accuracy()
        return acc

    def _crossval_data_loading(self):
        self.x,self.y=self.dataset.load_data(self.mods())
        self.input_shape=list(self.dataset.input_shape())
        self._preprocess()
        fold=0
        for self.tx,self.ty,self.vx,self.vy,self.gx,self.gy in crossvalidation(self.x,self.y,self.mods()):
            for obj in self.objs:
                obj.x=self.x
                obj.y=self.y
                obj.input_shape=self.input_shape
                obj.tx,obj.ty,obj.vx,obj.vy,obj.gx,obj.gy=self.tx,self.ty,self.vx,self.vy,self.gx,self.gy
                obj._preprocess_train()
            yield fold
            fold+=1

    def _create_embeddings(self):
        self.emb =np.concatenate([obj._embed(obj.tx) for obj in self.objs],axis=1)
        self.vemb=np.concatenate([obj._embed(obj.vx) for obj in self.objs],axis=1)
        self.gemb=np.concatenate([obj._embed(obj.gx) for obj in self.objs],axis=1)

    def __add__(self,other):
        if type(self) is ensemble and type(other) is ensemble:
            return ensemble(*self.objs,*other.objs)
        if type(self) is ensemble and type(other) is not ensemble:
            return self.add_objs(other)
        if type(self) is not ensemble and type(other) is ensemble:
            return other.add_objs(self)
        return ensemble(self,other)







