import numpy as np


myData = [0, 1, 0, 1, 0, 1, -300, 100, -300 , 100, -300, 100]
myData = np.array(myData)
print(myData)
reshapedData = np.reshape(myData,(-1, 2))
#reshapedData = np.reshape(myData)
print(reshapedData)