import eereid as ee

model_count=40                                                           #number of submodels
models=[ee.ghost(dataset=ee.datasets.market1501(),                       #create the submodels
                 model=ee.models.conv()) for i in range(model_count)]

model=ee.haunting(*models)                                               #build an ensemble

acc=model.evaluate()                                                     #Evaluate the approach
print(acc)



