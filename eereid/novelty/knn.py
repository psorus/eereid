from eereid.novelty.pyod_wrapper import pyod_wrapper
from pyod.models.knn import KNN


class knn(pyod_wrapper):

    def __init__(self,*args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__("knn")

    def init_model(self):
        self.model = KNN(*self.args, **self.kwargs)





