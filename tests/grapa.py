import eereid as ee


g=ee.ghost()

g(ee.datasets.mnist())
g(ee.models.graph())
g(ee.prepros.grapho(consider_count=100))
g(ee.prepros.subsample(0.001))

g["graph_flatten"]=False

acc=g.evaluate()

print(acc)


