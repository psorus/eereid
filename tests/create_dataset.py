#creates datasets:)
from eereid.datasets.from_folder import from_folder
import numpy as np

#function that takes a filename and returns the class label (as int)
label=lambda fn: int(fn.split("/")[-1].split("_")[0])

#function that takes a filename and returns True if the file should be included in the dataset
include=lambda fn: fn.endswith(".jpg") and not "/-1_" in fn

#files to possibly include (we dont differentiate between train and test data here)
files=["/home/psorus/d/test/data_goere/Market-1501-v15.09.15/bounding_box_train/*.jpg",
       "/home/psorus/d/test/data_goere/Market-1501-v15.09.15/bounding_box_test/*.jpg"]

dataset=from_folder(files,label,include)

x,y=dataset.load_raw()
np.savez_compressed("market1501.npz",x=x,y=y)







