import math
import numpy as np

import matplotlib.pyplot as plt

import BeeDNN as nn

print("Simple xor learning and classification")
print("Plot the Mean square error and a smooth decision surface")

# create train data
sample=np.zeros((4,2))
sample[0,0]=0 ; sample[1,0]=0 ; sample[2,0]=1 ; sample[3,0]=1
sample[0,1]=0 ; sample[1,1]=1 ; sample[2,1]=0 ; sample[3,1]=1
truth=np.zeros((4,1))
truth[0,0]=0 ; truth[1,0]=1  ; truth[2,0]=1 ; truth[3,0]=0

# construct net
n = nn.Net()
#n.append(nn.LayerAddUniformNoise()) # uncomment this to reduce overfitting
n.append(nn.LayerDense(2,3))
n.append(nn.LayerTanh())
n.append(nn.LayerDense(3,1))

# optimize net
train = nn.NetTrain()
train.epochs = 200
train.batch_size=0 #if set to zero, use full batch
train.set_optimizer(nn.OptimizerRPROPm())
train.set_loss(nn.LossMSE())
train.train(n,sample,truth)

# plot loss
plt.plot(train.epoch_loss)
plt.grid()
plt.title('Mean Square error vs. epochs')
plt.show(block=False)

# plot predicted smooth surface (without the last stairs step)
plt.figure()
x=np.linspace(-0.5,1.5,100) #extrapole +/-0.5
y=np.linspace(-0.5,1.5,100) #extrapole +/-0.5
X,Y=np.meshgrid(x,y)
Xr=np.atleast_2d(np.ravel(X)).transpose()
Yr=np.atleast_2d(np.ravel(Y)).transpose()
XY=np.concatenate((Xr,Yr),axis=1)
n.classification_mode=False # to show smooth decision surface (not stairs) 
Z=n.forward(XY)
Z=Z.reshape(X.shape)
plt.pcolormesh(X,Y,Z)
plt.title("Smooth decision surface")
plt.colorbar()
plt.grid()
plt.show()