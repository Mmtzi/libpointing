from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers.recurrent import GRU, LSTM
from numpy import genfromtxt
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
import os

epochs=10

# load Data
loaddata = genfromtxt('dataTest.csv', delimiter=',', skip_header=1)
#first X rows
data = loaddata[:2000,]
#print(data)
#scale to -1,1
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(data)
scalarY.fit(data.reshape(4000,1))
X = scalarX.transform(data)
print(X)
#ydata
dataY=data[:1981,1]
print(dataY)
y = scalarY.transform(dataY.reshape(1981,1))
print(y)
# define and fit the final model


def create_dataset(dataset, look_back):
  dataX = []
  for i in range(len(dataset)-look_back+1):
    a = dataset[i:(i+look_back), :]
    dataX.append(a)
  return np.array(dataX)

xTrain = create_dataset(X, 20)
yTrain= y


if os.path.exists('models/easy_lr_model.h5'):
    model = load_model('models/easy_lr_model.h5')
    print("loaded model")
else:
    model = Sequential()
    model.add(LSTM(4, return_sequences=True, input_shape=(20, 2), activation='relu'))
    model.add(LSTM(4))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    print("created model")
model.get_weights()
model.fit(xTrain, yTrain, epochs=epochs, verbose=2, batch_size=5)
model.save('models\\easy_lr_model.h5')
#data to test the model
Xnew = loaddata[2000:2500]
ynew= Xnew[:,1]
Xnew = scalarX.transform(Xnew)
# make a prediction
xTest = create_dataset(Xnew, 20)

ynew = model.predict(xTest)
# show the inputs and predicted outputs
ynew = scalarY.inverse_transform(ynew)
Xnew = scalarX.inverse_transform(Xnew)
onlyX =[]
for i in range(len(ynew)):

    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
    onlyX.append(Xnew[i][0])

fig, ax = plt.subplots()
print(len(onlyX), len(ynew))
ax.scatter(*zip(*data), color="b")
ax.scatter(onlyX, ynew, color="r")
i=1
while os.path.exists('logs\\epochs_'+str(epochs)+'-Iterations_'+str(i)+'.png'):
    i+=1
else:
    plt.savefig('logs\\epochs_'+str(epochs)+'-Iterations_'+str(i)+'.png')

plt.show()