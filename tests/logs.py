import eereid as ee

g=ee.ghost(dataset=ee.datasets.mnist(),
           distance=ee.distances.lN(2),
           loss=ee.losses.extended_triplet(),
           model=ee.models.conv(),
           preprocessing=ee.prepros.subsample(0.1),
           novelty=ee.novelty.knn(),
           triplet_count=100)

g["log_file"]="mnist_conv_lN2_triplet_knn_100.log"
g["log_level_print"]=0

acc=g.evaluate()
print(acc)

