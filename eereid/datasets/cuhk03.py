from eereid.datasets.downloadable import downloadable

class cuhk03(downloadable):
    def __init__(self):
        super().__init__('cuhk03')

    def explain(self):
        return "CUHK03 (Chinese University of Hong Kong Re-identification) data loader"


