import eereid as ee

g=ee.ghost(model=ee.models.conv(),
           dataset=ee.datasets.mnist(),
           loss=ee.losses.extended_triplet(),
           novelty=ee.novelty.distance(),
           preproc=ee.prepros.subsample(0.1),
           triplet_count=100,
           crossval=True)

acc=g.evaluate()

print(acc)


