from eereid.novelty.novelty import novelty


class pyod_wrapper(novelty):

    def __init__(self,name):
        super().__init__(name)
        self.init_model()

    def init_model(self):
        raise NotImplementedError

    def create_model(self,normal):
        self.model.fit(normal)

    def predict(self,samples):
        return self.model.decision_function(samples)

    def explain(self):
        return "Generic PyOD Wrapper Novelty Detection gag"




