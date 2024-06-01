import eereid as ee

model=ee.models.load_model("testmodel")

g=ee.ghost(dataset=ee.datasets.mnist(),
           distance=ee.distances.lN(2),
           loss=ee.losses.extended_triplet(),
           preprocessing=ee.prepros.subsample(0.1),
           novelty=ee.novelty.knn(),
           triplet_count=100)

g.load_model("testmodel")


acc=g.evaluate()

print(acc)

