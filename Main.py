import random
import numpy as np
import matplotlib.pyplot as plt
from ColorSpace import colorSpace as cp

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys         
sys.path.append(r'Algorithms') 

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

modelVal = net(n_entradas=1)
modelVal.load_state_dict(torch.load('./LoLi-IEA_V1/Models/modValVer3.pt'))
modelVal.eval()

modelSat = net(n_entradas=1)
modelSat.load_state_dict(torch.load('./LoLi-IEA_V1/Models/modSatVer4.pt'))
modelSat.eval()

def Enhancement(imageRGB):
    rows, columns, dimension = imageRGB.shape

    HSVInput= cp.RgbToHsv(imageRGB)

    #Component Division Low-Light
    hueComponent = HSVInput[:, :, 0]              
    satComponent = HSVInput[:, :, 1]
    valComponent = HSVInput[:, :, 2]

    valComponent1D =  valComponent.reshape([rows*columns,1])
    satComponent1D =  satComponent.reshape([rows*columns,1])

    valComponentTensor = torch.from_numpy(valComponent1D).float().to("cpu")
    satComponentTensor = torch.from_numpy(satComponent1D).float().to("cpu")

    valEnhancement = modelVal(valComponentTensor)
    valEnhComponent  =valEnhancement.detach().numpy().reshape([rows, columns])

    satEnhancement = modelSat(satComponentTensor)
    satEnhComponent  =satEnhancement.detach().numpy().reshape([rows, columns])

    HSV = np.dstack((hueComponent, satEnhComponent, valEnhComponent))
    algorithm = cp.HsvToRgb(HSV)

    return  algorithm

#Load Image
imagesLow = np.load('./ImageVector/low.npy', allow_pickle='TRUE')

RGBInput = []
algorithm = []
totalImage = 6
#imageQ = [286,267,166,350,39,155]
#imageQ = [155,32,166,420,44,25]
imageQ = [102,55,418,333,30,10]
for idx in range(totalImage):
    #  idxImage = random.randint(0, 485)
    #  RGBInput.append(imagesLow[idxImage])
    #  algorithm.append(Enhancement(imagesLow[idxImage]))
    #  print('Image Number:',idxImage) 
        
    RGBInput.append(imagesLow[imageQ[idx]])
    algorithm.append(Enhancement(imagesLow[imageQ[idx]]))
    print('Image Number:',imageQ[idx]) 

#np.save('imagesEnhancement.npy', algorithm)

rows = 3
columns = 4
imageIdx = 0
for idx in range(rows*columns):
    idx += 1
    if idx % 2 != 0:
        plt.subplot(rows,columns,idx)             
        plt.imshow(RGBInput[imageIdx])
        plt.title('ORIGINAL IMAGE')
        plt.axis('off')
    else:
        plt.subplot(rows,columns,idx)
        plt.imshow(algorithm[imageIdx])
        plt.title('ALGORITHM')
        plt.axis('off')
        imageIdx += 1
plt.show()


