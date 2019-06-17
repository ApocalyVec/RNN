

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
x_train = np.reshape(x_train, ())

# part 2 - building the RNN

# part3