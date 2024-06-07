import eereid as ee

g=ee.ghost(ee.prepros.subsample(0.001),ee.models.graph(),ee.prepros.grapho(), triplet_count=1000)

acc=g.evaluate()

print(acc)


