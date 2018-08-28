from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import losses, optimizers

import csv
from numpy import array
import pandas as pd
filename = 'dataTest.csv'
data = pd.read_csv(filename)
newData = array(data)
print(newData)

#loadData

dx = data.loc[:, :'dx']
rx = data.loc[:,'rx': ]
print(dx)
print(rx)
for i in dx:
    print(i)


#myMOdel
model = Sequential([
    Dense(100, input_dim=1, activation='relu'),
    Dense(1, activation='relu')
])
model.summary()

# Compile model
model.compile(loss=losses.mse, optimizer=optimizers.adam(lr=.0001,), metrics=['mse'])
# Fit the model

model.fit(dx, rx, epochs=100, validation_split=0.1, batch_size=10, verbose=2, shuffle=True)
# evaluate the model
#scores = model.evaluate(scaled_train_samples, rx)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
sampleArray = data.loc[:200, :'dx']
predictions = model.predict(sampleArray, batch_size=1, verbose=2)
for i in predictions:
    print(i)