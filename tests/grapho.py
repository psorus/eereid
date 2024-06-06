import eereid as ee

from plt import *

ds=ee.datasets.mnist()

x,y=ds.load_raw()

g=ee.prepros.grapho()

v=g._apply_one(x[0])

print(v)

g.show_transformation(x[0])
plt.show()


