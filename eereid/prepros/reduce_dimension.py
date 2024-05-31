from eereid.prepros.prepro import prepro

import cv2
import numpy as np

class reduce_dimension(prepro):
    def __init__(self, square_size, delta=None, func="avg"):
        self.size = size
        super().__init__("reduce_dimension")

    def _apply_one(self, image):#working on
        interpolation=cv2.INTER_AREA
        return cv2.resize(image, self.size,interpolation=interpolation)

    def _apply_special(self,eereid):
        eereid.input_shape[0]=self.size[0]
        eereid.input_shape[1]=self.size[1]

    def save(self,pth,index):
        super().save(pth,index,size=size)

