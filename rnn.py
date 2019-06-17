

# recurrent neural network

# part 1 - Data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Optimize the training process by applying Feature Scaling
# two ways of feature scaling: standardisation, normalization (minmax normalization)
# it is recommended that when using RNN, and having Sigmoid activation function, to use Normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 timesteps and 1 output
x_train = []
y_train = []

for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# part 2 - building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
# input shape = (time steps, number of indicators)
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))  # ignore 20% of the neurons in both forward and backward propagation

# Adding a second LSTM layer and some Dropout regularization
# we don't need to put input_shape here, because:
# 1. this is not the input layer, 2. the units = 50 from previous layer tells this layer what is its input_shape
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSM layer and some Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a forth LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50, return_sequences=False))  # or return_sequences is default at False
regressor.add(Dropout(0.2))

# Adding  the output layer
regressor.add(Dense(units=1))  # in this case, units is the dimension of the output layer

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(x_train, y_train, epochs=100, batch_size=32)

# saving the trained model
from keras.models import load_model

regressor.save('/trained_model/regresspr.h5')
'''
use this line if you want to load the model back
regressor = load_model('/trained_model/regressor.h5')
'''
# part3