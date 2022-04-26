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
from keras.models import load_model
from sklearn.model_selection import train_test_split


# returns: X (norm), y (norm), y_mean (orig), y_std (orig)
def get_X_y(data, window=10):
    norm_data = normalize(data)
    np_data = norm_data.to_numpy()
    X = []
    for i in range(len(np_data) - window):
        X.append([r for r in np_data[i:i+window]])
    return np.array(X), np.array(norm_data['Close'][window:]), np.array(data['Close'][window:]).mean(), np.array(data['Close'][window:]).std()


# display predicted vs true line graph
def plot_predicted_vs_true(y_pred, y_true):
    plt.clf()
    plt.plot(y_pred, label='predicted')
    plt.plot(y_true, label='true')
    plt.legend()
    plt.show()


# display parity plot
def plot_parity(y_pred, y_true):
    pass


# column-wise normalization of df
def normalize(data):
    return (data - data.mean()) / data.std()


# de-normalization of df using mean and std from original dataset
def denormalize(data, mean, std):
    return data * std + mean


def main():
    data = yf.download('SPY', '1980-01-01', '2022-01-01')
    # data = yf.download('BTC-USD', '2010-01-01', '2022-04-01')
    # data2 = data2.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
    # data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    # data.set_index('Date', inplace=True)
    data = data.drop(['Adj Close'], axis=1)
    print(data.head(10))

    # split into train/val/test sets
    num_samples = data.shape[0]
    cutoff1 = math.floor(0.7 * num_samples)
    cutoff2 = math.floor(0.9 * num_samples)
    train_data = data[:cutoff1]
    val_data = data[cutoff1:cutoff2]
    test_data = data[cutoff2:]

    # normalize data
    X_train, y_train, _, _ = get_X_y(train_data)
    X_val, y_val, _, _ = get_X_y(val_data)
    X_test, y_test, test_mean, test_std = get_X_y(test_data)

    # build CNN-LSTM network
    model = Sequential()
    model.add(InputLayer((10, 5)))
    # model.add(Conv1D(64, kernel_size=2, activation='relu'))
    # model.add(Flatten())
    model.add(LSTM(64, activation='softsign'))
    #model.add(Dense(8, 'relu'))
    #model.add(LSTM(5, activation='softsign'))
    model.add(Dense(1, 'linear'))
    print(model.summary())

    # model = tf.keras.Sequential([
    #     InputLayer((10, 1)),
    #     LSTM(5, return_sequences=True, activation='softsign'),
    #     Dropout(0.10),
    #     LSTM(50, return_sequences=True, activation='softsign'),
    #     Dropout(0.10),
    #     LSTM(20),
    #     Dense(1, 'linear')
    # ])
    # model.build()
    # print(model.summary())

    cp = keras.callbacks.ModelCheckpoint('model/',
                                         save_best_only=True)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                   metrics=[RootMeanSquaredError()])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[cp])

    # y_pred = denormalize(pd.DataFrame(model.predict(X).flatten()), mean, std).to_numpy()
    y_pred = denormalize(pd.DataFrame(model.predict(X_test).flatten()), test_mean, test_std).to_numpy()
    y_test = denormalize(pd.DataFrame(y_test), test_mean, test_std).to_numpy()
    plot_predicted_vs_true(y_pred, y_test)


if __name__ == '__main__':
    main()
