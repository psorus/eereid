from eereid.datasets.downloadable import downloadable

class metal(downloadable):
    def __init__(self):
        super().__init__('metal')

    def explain(self):
        return "Metal block data loader"


