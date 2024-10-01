import eereid as ee
import numpy as np
from plt import *


g=ee.ghost(ee.datasets.market1501(),ee.prepros.classsample(0.1),triplet_count=100,epochs=1)

acc=g.evaluate()

g.plot_matches(0,n=5)

plt.show()








