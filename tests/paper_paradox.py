import eereid as ee

model_count=4
models=[ee.ghost(ee.prepros.subsample(0.1), triplet_count=1000) for i in range(model_count)]
model=ee.haunting(*models)

acc=model.evaluate()

print(acc)



