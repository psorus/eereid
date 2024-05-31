from eereid.novelty.pyod_wrapper import pyod_wrapper
from pyod.models.sod import SOD


class sod(pyod_wrapper):

    def __init__(self,*args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__("sod")

    def init_model(self):
        self.model = SOD(*self.args, **self.kwargs)





