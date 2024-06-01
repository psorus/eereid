from eereid.novelty.pyod_wrapper import pyod_wrapper
try:
    from pyod.models.knn import KNN
except ImportError:
    from eereid.importhelper import importhelper
    KNN=importhelper("pyod","knn novelty")


class knn(pyod_wrapper):

    def __init__(self,*args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__("knn")

    def init_model(self):
        self.model = KNN(*self.args, **self.kwargs)





