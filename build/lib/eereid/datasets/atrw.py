from eereid.datasets.downloadable import downloadable

class atrw(downloadable):
    def __init__(self):
        super().__init__('atrw')

    def explain(self):
        return "ATRW (tiger) data loader"


