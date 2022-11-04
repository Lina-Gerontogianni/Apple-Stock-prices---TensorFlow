## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import tensorflow as tf

# for reproducibility purpose 
from numpy.random import seed
seed(42)
tf.random.set_seed(42)
sk.utils.check_random_state(42)

## Cleaning data
dataset = pd.read_csv('AAPL.csv', index_col=0)
dataset = dataset.iloc[:,1:7]
dataset['date'] = dataset['date'].map(lambda x: x.rstrip(' 00:00:00+00:00'))

## Extracting the 2020 stock proces as test data
test_data_allfeat = dataset[dataset['date'].str.contains('2020')].to_csv('test_data.csv',index=False)
test_data_allfeat = pd.read_csv('test_data.csv')
test_data = test_data_allfeat.iloc[:,4:5].values

# Removing the 2020 stock prices from the training data
train_data_allfeat = dataset[~dataset['date'].str.contains('2020')]
train_data = train_data_allfeat.iloc[:,4:5].values

## Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
train_data_scaled = scaler.fit_transform(train_data)

## Splitting to training and test data
X_train = []
y_train = []

steps = 50 # number of past stock prices to use for future stock price prediction 
for t in range(50,train_data.shape[0]):
    X_train.append(train_data_scaled[t - steps:t, 0])
    y_train.append(train_data_scaled[t, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

## Importing the libraries for the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

## Initialising the RNN
lstm = Sequential()

## Adding the 1st LSTM layer and some Dropout regularisation
lstm.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
lstm.add(Dropout(0.3))

## Adding extra LSTM layer and some Dropout regularisation
lstm.add(LSTM(units = 50, return_sequences = True))
lstm.add(Dropout(0.3))

## Adding extra LSTM layer and some Dropout regularisation
lstm.add(LSTM(units = 50, return_sequences = True))
lstm.add(Dropout(0.3))

## Adding extra LSTM layer and some Dropout regularisation
lstm.add(LSTM(units = 20))
lstm.add(Dropout(0.2))

#3 Adding the output layer
lstm.add(Dense(units = 1))

## Compiling the RNN
lstm.compile(optimizer = tf.keras.optimizers.Adam(lr=0.01), loss = 'mean_squared_error')

## Fitting the RNN to the Training set
lstm.fit(X_train, y_train, epochs = 60, batch_size = 30)


## Getting the predicted stock price of 2020
dataset_total = pd.concat((train_data_allfeat['open'], test_data_allfeat['open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_data) - steps:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(steps, steps+test_data.shape[0]):
    X_test.append(inputs[i-steps:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = lstm.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(test_data, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted')
plt.title('Apple Stock Price 2020 Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
