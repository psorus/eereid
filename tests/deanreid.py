import eereid as ee
import time



model_count=200
models=[]
for i in range(model_count):
    model=ee.ghost(
                #ee.datasets.market1501(),
                ee.datasets.palletlight(),
                #ee.prepros.subsample(0.1), 
                ee.prepros.featurebagging(100),
                ee.models.simple_dense(),
                #ee.prepros.trainsample(0.1),
                triplet_count=1000,
                output_size=10,
                )
    models.append(model)



model=ee.haunting(*models)

t0=time.time()
acc=model.evaluate()
t1=time.time()

print(acc)
print(t1-t0)


