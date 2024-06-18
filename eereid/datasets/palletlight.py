from eereid.datasets.downloadable import downloadable

class palletlight(downloadable):
    def __init__(self):
        super().__init__('spalletlight')

    def explain(self):
        return "Pallet block 502 data loader, only using images with the light turned on"


