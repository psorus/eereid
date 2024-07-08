from eereid.prepros.prepro import prepro

try:
    import cv2
except ImportError:
    from eereid.importhelper import importhelper
    cv2=importhelper("cv2","resize","pip install opencv-python")
import numpy as np

class crop(prepro):
    def __init__(self, topx=0,topy=0,botx=0,boty=0):
        self.topx=topx
        self.topy=topy
        self.botx=botx
        self.boty=boty
        super().__init__("crop")

    def _apply_one(self, image):
        return image[self.topy:image.shape[0]-self.boty,self.topx:image.shape[1]-self.botx]

    def _apply_special(self,eereid):
        eereid.input_shape[0]=eereid.input_shape[0]-self.topy-self.boty
        eereid.input_shape[1]=eereid.input_shape[1]-self.topx-self.botx

    def save(self,pth,index):
        super().save(pth,index,topx=self.topx,topy=self.topy,botx=self.botx,boty=self.boty)

    def stage(self):return "general"
    def order(self):return 3

    def explain(self):
        return f"Resizing each image to a size of {self.size}"
