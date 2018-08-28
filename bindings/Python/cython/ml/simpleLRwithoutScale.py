from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers.recurrent import LSTM
import numpy as np
from numpy import genfromtxt
from keras import regularizers
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import os

def create_dataset(dataset, look_back):
  dataX = []
  for i in range(len(dataset)-look_back+1):
    a = dataset[i:(i+look_back)]
    dataX.append(a)
    print(a)
  return np.array(dataX)


epochs=1000

# load Data
loaddata = genfromtxt('dataTest.csv', delimiter=',', skip_header=1)
#first X rows
inputData = loaddata[:2000,0]
fulloutPutData = loaddata[:2000,1]
print(inputData)
#scale to -1,1

#ydata
outputData=loaddata[:1981,1]
print(outputData)

xTrain = create_dataset(inputData, 20)
yTrain = outputData

# define and fit the final model
if os.path.exists('models/unscale_easy_lr_model.h5'):
    model = load_model('models/unscale_easy_lr_model.h5')
    print("loaded model")
else:
    model = Sequential()
    #model.add(LSTM(4, return_sequences=True, input_shape=(20, 1), activation='relu'))
    #model.add(LSTM(4))
    model.add(Dense(8, input_dim=20, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    print("created model")
model.get_weights()
#fit generator läd daten für einen batch np samples 128 batch size
model.fit(xTrain, yTrain, epochs=epochs, verbose=2, batch_size=5, validation_split=0.05)
model.save('models\\unscale_easy_lr_model.h5')
#data to test the model
testInput = loaddata[2000:2500,0]
testData = create_dataset(testInput,20)
testLabels = loaddata[2000:2500,1]
# make a prediction
testOutput = model.predict(testData)
# show the inputs and predicted outputs

for i in range(len(testData)):

    print("X=%s, Predicted=%s ,Original=%s"  % (testInput[i], testOutput[i], testLabels[i]))

fig, ax = plt.subplots()
print(len(testInput), len(testOutput))
ax.scatter(inputData, fulloutPutData, color="b")
ax.scatter(testInput[:481], testOutput, color="r")
#functionInput = np.arange(-30,30,   20)
#unctionOutput = model.predict(testData)
#ax.scatter(testData, functionOutput, color="k", marker='+')
i=1
while os.path.exists('logs\\epochs_'+str(epochs)+'-Iterations_'+str(i)+'.png'):
    i+=1
else:
    plt.savefig('logs\\epochs_'+str(epochs)+'-Iterations_'+str(i)+'.png')

plt.show()