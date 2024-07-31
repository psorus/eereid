from eereid.prepros.prepro import prepro

import numpy as np

from tqdm import tqdm

class apply_func(prepro):
    #applies a function, to each part of the image, and stiches the resulting values back together
    def __init__(self, func=None, subimagesize=4, overlap=0.5):
        if func is None:
            def func(x):
                x=np.mean(x,axis=1)
                x=np.mean(x,axis=1)
                return x
        self.func = func
        self.subimagesize = subimagesize
        self.overlap = overlap
        super().__init__("apply_func")


    def apply(self, data, labels, eereid):
        if eereid is not None:self._apply_special(eereid)
        data= np.array([self._apply_one(image) for image in tqdm(data,desc="Applying function to data")])
        if eereid is not None:eereid.input_shape=data[0].shape

        return data, labels


    def _apply_one(self, image):#working on
        delta=int(self.subimagesize*(1-self.overlap))
        subimages=[]
        dim1=0
        for i in range(0,image.shape[0]-self.subimagesize,delta):
            dim1+=1
            dim2=0
            for j in range(0,image.shape[1]-self.subimagesize,delta):
                dim2+=1
                subimages.append(image[i:i+self.subimagesize,j:j+self.subimagesize])
        subimages=np.array(subimages)
        values=self.func(subimages)
        newimage=np.reshape(values,(dim1,dim2,-1))
        return newimage

    def _apply_special(self,eereid):
        pass

    def save(self,pth,index):
        super().save(pth,index,size=size)

    def stage(self):return "general"
    def order(self):return 1

    def explain(self):
        return f"Applies a function to each subimage and stiches the result back into a new image. Uses subimages of {self.subimagesize} pixels in each direction with overlap of {self.overlap}"

