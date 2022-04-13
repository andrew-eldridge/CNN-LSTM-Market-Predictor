import math

import keras.callbacks
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from keras.metrics import RootMeanSquaredError
from sklearn.model_selection import train_test_split


def get_X_y(data, window=10):
    np_data = data.to_numpy()
    X = []
    y = []
    for i in range(len(np_data) - window):
        X.append([r for r in np_data[i:i+window]])
        y.append(np_data[i+window][3])
        # print(X)
        # print(y)
        # exit(0)
    return np.array(X), np.array(y)


def main():
    data = yf.download('SPY', '1980-01-01', '2022-01-01')
    # data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    # data.set_index('Date', inplace=True)
    data = data.drop(['Adj Close', 'Volume'], axis=1)
    print(data.head(10))
    # data['Open'].plot()
    # plt.show()

    X, y = get_X_y(data)
    print(X[:10])
    print(y[:10])

    num_samples = len(y)
    cutoff_1 = math.floor(0.7*num_samples)
    cutoff_2 = math.floor(0.85*num_samples)
    X_train = X[:cutoff_1]
    y_train = y[:cutoff_1]
    X_val = X[cutoff_1:cutoff_2]
    y_val = y[cutoff_1:cutoff_2]
    X_test = X[cutoff_2:]
    y_test = y[cutoff_2:]

    # plt.plot(y_train)
    # plt.plot(y_val)
    # plt.plot(y_test)
    # plt.show()

    # X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, train_size=0.7)
    # X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, train_size=0.5)

    # print(X_train.shape)
    # print(y_train.shape)
    #
    # print(X_test.shape)
    # print(y_test.shape)
    #
    # print(X_val.shape)
    # print(y_val.shape)
    #
    # exit(0)

    model = Sequential()
    model.add(InputLayer((10, 4)))
    # model.add(Conv1D(64, kernel_size=2))
    model.add(LSTM(64))
    # model.add(Flatten())
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))

    cp = keras.callbacks.ModelCheckpoint('model/',
                                         save_best_only=True)
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                  metrics=[RootMeanSquaredError()])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[cp])
    y_pred = model.predict(X_test).flatten()
    plt.plot(y_pred)
    plt.plot(y_train)
    plt.plot(y_val)
    plt.plot(y_test)
    plt.show()


if __name__ == '__main__':
    main()
