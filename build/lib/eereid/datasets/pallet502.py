from eereid.datasets.downloadable import downloadable

class pallet502(downloadable):
    def __init__(self):
        super().__init__('spallet502')

    def explain(self):
        return "Pallet block 502 data loader"


