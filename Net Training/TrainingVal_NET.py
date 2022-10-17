import numpy as np
import pandas as pd
import imageio
import sys         
sys.path.append(r'Algorithm Ver 1\ColorSpace') 
from ColorSpace   import colorSpace       as cp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


RGBlowTrain  = np.load('low.npy', allow_pickle='TRUE')[1]
RGBhighTrain = np.load('high.npy', allow_pickle='TRUE')[1]

rows, columns, dimension = RGBlowTrain.shape

HSVlowTrain= cp.RgbToHsv(RGBlowTrain)
HSVhighTrain = cp.RgbToHsv(RGBhighTrain)

#Component Division Low-Light
HUElowTrain = HSVlowTrain[:, :, 0]              
SATlowTrain = HSVlowTrain[:, :, 1]
VALlowTrain = HSVlowTrain[:, :, 2]


#Component Division High-Light
HUEhighTrain = HSVhighTrain[:, :, 0]              
SAThighTrain = HSVhighTrain[:, :, 1]
VALhighTrain = HSVhighTrain[:, :, 2]


#2DTo1D reshape([1,RowsxColumns])
# VALlow1DTrain =  VALlowTrain.reshape([rows*columns,1])
# VALlhigh1DTrain =  VALhighTrain.reshape([rows*columns,1])

VALlow1DTrain =    np.load('xValv3.npy', allow_pickle='TRUE')
VALlhigh1DTrain =  np.load('yValv3.npy', allow_pickle='TRUE')

t_X_train = torch.from_numpy(VALlow1DTrain).float().to("cpu")
t_y_train = torch.from_numpy(VALlhigh1DTrain).float().to("cpu")


class net(nn.Module):

    def __init__(self, n_entradas):
        super(net, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 15)
        self.linear2 = nn.Linear(15, 8) 
        self.linear3 = nn.Linear(8, 1)

    def forward(self, inputs):
        pred_1 = torch.relu(input=self.linear1(inputs))
        pred_2 = torch.relu(input=self.linear2(pred_1))
        pred_f = torch.relu(input=self.linear3(pred_2))
        return pred_f

lr = 0.001
epochs = 2000
estatus_print = 100

model = net(n_entradas=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
print("Model architecture: {}".format(model))
historico = pd.DataFrame()


print("Training the Model")
for epoch in range(1,epochs+1):
    y_pred = model(t_X_train)
    loss = loss_fn(input=y_pred, target=t_y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % estatus_print == 0:
        print(f"\nEpoch {epoch}\t Loss: {round(loss.item(), 4)}")

yVALTrain = model(t_X_train)
PATH = './modValVer3.pt'
torch.save(model.state_dict(), PATH)

yVALTTrain =yVALTrain.detach().numpy().reshape([rows, columns])
#HSV = np.dstack((HUElowTrain,  yVALTTrain, VALhighTrain))
HSV = np.dstack((HUElowTrain,  SAThighTrain, yVALTTrain))
algorithm = cp.HsvToRgb(HSV)


plt.subplot(1,3,1)             
plt.imshow(RGBlowTrain)#, cmap='gray')
plt.title('ORIGINAL IMAGE')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(algorithm)#, cmap='gray')
plt.title('ALGORITHM')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(RGBhighTrain)#, cmap='gray')
plt.title('TARGET')
plt.axis('off')
plt.show()

##########################################################
# plt.figure(figsize=(10, 10))
# plt.plot(historico['Epoch'], historico['Loss'], label='Loss')
# plt.title("Loss", fontsize=25)
# plt.xlabel("Epoch", fontsize=12)
# plt.ylabel("Loss", fontsize=12)
# plt.grid()
# plt.show()

# plt.figure(figsize=(10, 10))
# plt.plot(historico['Epoch'], historico['Accuracy'], label='Accuracy')
# plt.title("Accuracy", fontsize=25)
# plt.xlabel("Epoch", fontsize=12)
# plt.ylabel("Accuracy", fontsize=12)
# plt.grid()
# plt.show()
