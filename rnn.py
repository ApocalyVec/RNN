

# recurrent neural network

# part 1 - Data preprocessing
import os
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

time_steps = 120

for i in range(time_steps, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-time_steps:i, 0])
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

regressor.save(os.path.join('trained_model', 'regressor_ts_120.h5'))
'''
use this line if you want to load the model back
regressor = load_model('/trained_model/regressor.h5')
'''
# part3
# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_steps:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(time_steps, len(inputs)):
    X_test.append(inputs[i-time_steps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)  # making the prediction
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
