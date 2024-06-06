import eereid as ee


g=ee.ghost()

g(ee.datasets.mnist())
g(ee.models.graph())
g(ee.prepros.grapho())
g(ee.prepros.subsample(0.001))

acc=g.evaluate()

print(acc)


