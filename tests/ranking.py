import eereid as ee
from plt import *

model=ee.models.load_model("testmodel")

g=ee.ghost(dataset=ee.datasets.mnist(),
           distance=ee.distances.lN(2),
           loss=ee.losses.extended_triplet(),
           preprocessing=ee.prepros.subsample(0.1),
           novelty=ee.novelty.knn(),
           triplet_count=100)

g.load_data("testdata.npz")
g.load_model("testmodel")


acc=g.evaluate()

print(acc)


i=0

g.plot_match(g.qx[i],g.qy[i],4)
plt.savefig("last.png")
plt.show()


