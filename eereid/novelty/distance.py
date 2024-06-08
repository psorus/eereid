from eereid.novelty.novelty import novelty

import numpy as np


class distance(novelty):

    def __init__(self,metric="auto"):
        super().__init__("distance novelty detection")

        self.metric=metric

    def inherit_info(self, ghost):
        if self.metric=="auto":
            self.metric=ghost.distance

    def create_model(self,normal):
        self.normal=normal

    def predict(self,samples):
        ret=np.array([np.min(self.metric.multi_distance(self.normal,sample)) for sample in samples])
        return ret

    def explain(self):
        return "Novelty detection based on distance from the normal data. Using the following distance function:\n    "+self.metric.explain()




