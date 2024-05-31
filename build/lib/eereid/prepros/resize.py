from eereid.prepros.prepro import prepro

import cv2
import numpy as np

class resize(prepro):
    def __init__(self, size):
        self.size = size
        super().__init__("resize")

    def _apply_one(self, image):
        interpolation=cv2.INTER_AREA
        return cv2.resize(image, self.size,interpolation=interpolation)

    def _apply_special(self,eereid):
        eereid.input_shape[0]=self.size[0]
        eereid.input_shape[1]=self.size[1]

    def save(self,pth,index):
        super().save(pth,index,size=size)

