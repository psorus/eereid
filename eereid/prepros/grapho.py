from eereid.prepros.prepro import prepro
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from tqdm import tqdm

class grapho(prepro):
    def __init__(self,nodes=50,func=None,subimagesize=2, consider_count=250, connections=3):
        self.nodes=nodes
        if func is None:
            def func(x):
                x=np.array(x)
                x=np.abs(x)
                while len(x.shape)>1:
                    x=np.mean(x,axis=-1)
                return x
        self.func=func
        self.subimagesize=subimagesize
        self.consider_count=consider_count
        self.connections=connections
        super().__init__("grapho")

    def apply(self, data, labels, eereid):
        if eereid is not None:self._apply_special(eereid)
        data= np.array([self._apply_one(image) for image in tqdm(data)])
        if eereid is not None:eereid.input_shape=data[0].shape

        return data, labels

    def _apply_one(self, image):
        #transforms an image into a graph of shape (nodes,nodes+features)
        #first iterate over all subimages and evaluate func. Find the highest consider_count subimages and their position

        positions=[]
        subimages=[]
        for i in range(image.shape[0]-self.subimagesize):
            for j in range(image.shape[1]-self.subimagesize):
                subimage=image[i:i+self.subimagesize,j:j+self.subimagesize]
                subimages.append(subimage)
                positions.append((i+self.subimagesize/2,j+self.subimagesize/2))

        values=self.func(subimages)
        border=np.sort(values)[-self.consider_count]
        positions=np.array(positions)[values>border]
        values=values[values>border]

        #now lets combine similar positions together, until we have nodes positions left

        kmeans=KMeans(n_clusters=self.nodes)
        kmeans.fit(positions)
        centers=kmeans.cluster_centers_

        #build a topk graph

        graph=np.zeros((self.nodes,self.nodes))
        for i, center in enumerate(centers):
            distances=np.linalg.norm(centers-center,axis=1)
            topk=np.argsort(distances)[:self.connections]
            for j in topk:
                graph[i,j]=1
                graph[j,i]=1


        #build some features
        #pos x, pos y, avg color in subimages, std color in subimages, subimages in cluster

    
        cluster_to_regions=[]
        labels=kmeans.labels_
        for i in range(self.nodes):
            cluster_to_regions.append(np.where(labels==i)[0])

        pos=centers
        subimagecount=np.array([[len(x)] for x in cluster_to_regions])
        def get_all_pixel(indices):
            pixels=[]
            for i in indices:
                x,y=positions[i]
                x=int(x-self.subimagesize/2)
                y=int(y-self.subimagesize/2)
                pixels.append(image[x:x+self.subimagesize,y:y+self.subimagesize])
            pixels=np.array(pixels)
            return np.reshape(pixels,(pixels.shape[0]*pixels.shape[1]*pixels.shape[2],-1))
        means,stds=[],[]
        for region in cluster_to_regions:
            pixels=get_all_pixel(region)
            means.append(np.mean(pixels,axis=0))
            stds.append(np.std(pixels,axis=0))
        means=np.array(means)
        stds=np.array(stds)
        features=np.concatenate([pos,means,stds,subimagecount],axis=1)


        data=np.concatenate([graph,features],axis=1)

        return data


    def show_transformation(self, image):
        data=self._apply_one(image)
        graph=data[:,:self.nodes]
        features=data[:,self.nodes:]
        centers=features[:,:2]
        plt.imshow(image,cmap="hot")
        colo="green"
        plt.scatter(centers[:,1],centers[:,0],c=colo)

        for i in range(self.nodes):
            for j in range(i+1,self.nodes):
                if graph[i,j]==1:
                    plt.plot([centers[i,1],centers[j,1]],[centers[i,0],centers[j,0]],c=colo)


    def _apply_special(self,eereid):
        pass

    def stage(self):return "general"
    def order(self):return 10

