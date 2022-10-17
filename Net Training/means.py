import numpy as np
import sys         
sys.path.append(r'Algorithm Ver 1\ColorSpace') 
from ColorSpace   import colorSpace       as cp
import matplotlib.pyplot as plt

lowSet  = np.load('./ImageVector/low.npy', allow_pickle='TRUE')
highSet = np.load('./ImageVector/high.npy', allow_pickle='TRUE')

lowSet = lowSet[1:3]
highSet = highSet[1:3]

def loadImage(idx,x,y):
    RGBlowTrain  = x[idx]
    RGBhighTrain = y[idx]

    rows, columns, dimension = RGBlowTrain.shape

    HSVlowTrain= cp.RgbToHsv(RGBlowTrain)
    HSVhighTrain = cp.RgbToHsv(RGBhighTrain)

    #Component Division Low-Light            
    SATlowTrain = HSVlowTrain[:, :, 1]
    VALlowTrain = HSVlowTrain[:, :, 2]


    #Component Division High-Light          
    SAThighTrain = HSVhighTrain[:, :, 1]
    VALhighTrain = HSVhighTrain[:, :, 2]


    #2DTo1D reshape([1,RowsxColumns])
    SATlow1DTrain   =  SATlowTrain.reshape([rows*columns,1])
    SATlhigh1DTrain =  SAThighTrain.reshape([rows*columns,1])
    VALlow1DTrain   =  VALlowTrain.reshape([rows*columns,1])
    VALlhigh1DTrain =  VALhighTrain.reshape([rows*columns,1])

    #return VALlow1DTrain , VALlhigh1DTrain
    return SATlow1DTrain, SATlhigh1DTrain

vectoresLow = []
vectoresHigh = []

for i in range(len(lowSet)):
    imagelow, imagehigh = loadImage(i,lowSet,highSet)
    vectoresLow.append(imagelow)
    vectoresHigh.append(imagehigh)

vectoresLow  = np.array(vectoresLow)
vectoresHigh = np.array(vectoresHigh)


rows, columns, dimensions = vectoresLow.shape

arrayLow  = []
meanLow   = []
arrayHigh = []
meanHigh  = []

for i in range(columns):
    for j in range(rows):
        arrayLow.append(vectoresLow[j][i])
        arrayHigh.append(vectoresHigh[j][i])
    meanLow.append(np.mean(arrayLow))
    meanHigh.append(np.mean(arrayHigh))
    arrayLow  = []
    arrayHigh = []

meanLow = np.array(meanLow)
meanHigh = np.array(meanHigh)

meanLow  = meanLow[:,None]
meanHigh = meanHigh[:,None]
meanHigh += 0.1

print("Mean Low:",  meanLow.shape)
print("Mean High:", meanHigh.shape)

# np.save('xValv3.npy', meanLow)
# np.save('yValv3.npy', meanHigh)
np.save('xSATv33.npy', meanLow)
np.save('ySATv33.npy', meanHigh)


print("Low:")
print(vectoresLow[0][0])
print(vectoresLow[1][0])
#print(vectoresLow[2][0])
print("Mean Low:",meanLow[0])

print("High:")
print(vectoresHigh[0][0])
print(vectoresHigh[1][0])
#print(vectoresHigh[2][0])
print("Mean High:",meanHigh[0])


plt.subplot(1,2,1)             
plt.imshow(highSet[0])
plt.title('IMAGE')
plt.axis('off')
plt.subplot(1,2,2)             
plt.imshow(highSet[1])
plt.title('IMAGE')
plt.axis('off')
plt.show()