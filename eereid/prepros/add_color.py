from eereid.prepros.prepro import prepro
import numpy as np

class add_color(prepro):
    def __init__(self):
        super().__init__("add_color")

    def apply(self, data, labels, eereid):
        self._apply_special(eereid)
        data=np.expand_dims(data,axis=-1)
        data=np.repeat(data,3,axis=-1)
        return data, labels

    def _apply_one(self, image):
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)
        return image

    def _apply_special(self,eereid):
        eereid.input_shape=list(eereid.input_shape)+[3]

    def save(self,pth,index):
        super().save(pth,index,r=self.r,g=self.g,b=self.b)

    def stage(self):return "general"
    def order(self):return 4

    def explain(self):
        return "Preprocessing that adds a color channel to the data by repeating every value 3 times. This is useful when the model expects a 3-channel input, but the data is grayscale."

