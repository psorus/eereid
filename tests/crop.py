import eereid as ee

g=ee.ghost(dataset=ee.datasets.mnist(),
           distance=ee.distances.lN(2),
           loss=ee.losses.extended_triplet(),
           model=ee.models.conv(),
           preprocessing=ee.prepros.subsample(0.1),
           triplet_count=100)

g(ee.prepros.crop(5,5,5,5))


acc=g.evaluate()
print(acc)

