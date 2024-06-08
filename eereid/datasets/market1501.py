from eereid.datasets.downloadable import downloadable

class market1501(downloadable):
    def __init__(self):
        super().__init__('market1501')

    def explain(self):
        return "Market-1501 data loader"


