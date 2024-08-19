import eereid as ee
import time

import numpy as np

t0=time.time()

x,y=ee.datasets.palletlight().load_data()

##normal mode
tx,ty,qx,qy,gx,gy=ee.tools.datasplit(x,y,ee.ghost().mods())

##self mode
#classes=np.unique(y)
#c2x={cls:[] for cls in classes}
#for cls,xx in zip(y,x):
#    c2x[cls].append(xx)
#trainf=0.5
#tx,ty=[],[]
#qx,qy=[],[]
#for cls,xx in c2x.items():
#    np.random.shuffle(xx)
#    count=int(trainf*len(xx))
#    tx.extend(xx[:count])
#    ty.extend([cls]*count)
#    qx.extend(xx[count:])
#    qy.extend([cls]*(len(xx)-count))
#tx,ty,qx,qy=np.array(tx),np.array(ty),np.array(qx),np.array(qy)
#gx,gy=tx,ty




print(tx.shape,ty.shape,qx.shape,qy.shape,gx.shape,gy.shape)


q,g=[],[]


model_count=10
for i in range(model_count):
    model=ee.ghost(
                #ee.datasets.market1501(),
                ee.datasets.from_numpy(tx,ty),
                #ee.prepros.subsample(0.1), 
                ee.prepros.featurebagging(1000),
                #total=250*450*3=337500
                ee.models.simple_dense(),
                #ee.prepros.trainsample(0.1),
                triplet_count=2500,
                output_size=25,
                epochs=3,
                )
    model.train()
    aq,ag=model.embed(qx),model.embed(gx)
    q.append(aq)
    g.append(ag)

q=np.concatenate(q,axis=1)
g=np.concatenate(g,axis=1)

acc=ee.tools.rankN(q,qy,g,gy,distance=ee.distances.euclidean())
print(acc)

t1=time.time()
print(t1-t0)


