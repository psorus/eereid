#creates datasets:)
from eereid.datasets.from_folder import from_folder
import numpy as np

label=lambda fn: int(fn.split("/")[-1].split("_")[0])
include=lambda fn: fn.endswith(".jpg") and not "/-1_" in fn
dataset=from_folder(["/home/psorus/d/test/data_eereid/Market-1501-v15.09.15/bounding_box_train/*.jpg",
                    "/home/psorus/d/test/data_eereid/Market-1501-v15.09.15/bounding_box_test/*.jpg"],
                    label,include)

x,y=dataset.load_raw()
np.savez_compressed("market1501.npz",x=x,y=y)







