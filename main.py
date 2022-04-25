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
import os


def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)


def get_X_y(data, window=5):
    np_data = data.to_numpy()
    X = []
    y = []
    for i in range(len(np_data) - window):
        X.append([r for r in np_data[i:i+window]])
        y.append(np_data[i+window][0])
        # print(X)
        # print(y)
        # exit(0)
    return np.array(X), np.array(y)


def show_plots(y_pred, y_true):
    #model = load_model('model1/')
    #train_predictions = model.predict(X_train).flatten()
    #train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})

    plt.clf()
    plt.plot(y_pred[1000:1050], label='predicted')
    plt.plot(y_true[1000:1050], label='true')
    plt.legend()
    plt.show()


def normalize(data, mean, std):
    return (data - mean) / std


def denormalize(data, mean, std):
    return data * std + mean


def main():
    # zip_path = tf.keras.utils.get_file(
    #     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    #     fname='jena_climate_2009_2016.csv.zip',
    #     extract=True)
    # csv_path, _ = os.path.splitext(zip_path)
    # df = pd.read_csv(csv_path)
    # df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    # print(df.head(10))
    #
    # temp = df['T (degC)']
    # temp.plot()
    #
    # X1, y1 = df_to_X_y(temp)
    # X_train1, y_train1 = X1[:60000], y1[:60000]
    # X_val1, y_val1 = X1[60000:65000], y1[60000:65000]
    # X_test1, y_test1 = X1[65000:], y1[65000:]
    #
    # model1 = Sequential()
    # model1.add(InputLayer((5, 1)))
    # model1.add(LSTM(64))
    # model1.add(Dense(8, 'relu'))
    # model1.add(Dense(1, 'linear'))
    #
    # print(f'X_train: {X_train1[:10]}')
    # print(f'y_train: {y_train1[:10]}')
    #
    # cp1 = keras.callbacks.ModelCheckpoint('model1/', save_best_only=True)
    # model1.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    # model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=5, callbacks=[cp1])
    #
    # y_pred1 = model1.predict(X_test1).flatten()
    #
    # show_plots(y_pred1, y_test1)
    #
    # exit(0)

    data = yf.download('SPY', '1980-01-01', '2022-01-01')
    # data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    # data.set_index('Date', inplace=True)
    data = data.drop(['Adj Close', 'Volume', 'Open', 'High', 'Low'], axis=1)
    print(data.head(10))
    # data['Open'].plot()
    # plt.show()

    X, y = get_X_y(data)
    print(f'X head: {X[:10]}')
    print(f'y head: {y[:10]}')

    # partition data into train/val/test sets
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=0.7)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.5)
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)

    # disjoint partitions
    # num_samples = len(y)
    # cutoff_1 = math.floor(0.7*num_samples)
    # cutoff_2 = math.floor(0.85*num_samples)
    # X_train = X[:cutoff_1]
    # y_train = pd.DataFrame(y[:cutoff_1])
    # X_val = X[cutoff_1:cutoff_2]
    # y_val = pd.DataFrame(y[cutoff_1:cutoff_2])
    # X_test = X[cutoff_2:]
    # y_test = pd.DataFrame(y[cutoff_2:])

    # normalize target values
    mean = y.mean()
    std = y.std()
    y_train = normalize(y_train, mean, std)
    y_train = y_train.to_numpy()
    y_val = normalize(y_val, mean, std)
    y_val = y_val.to_numpy()
    y_test = normalize(y_test, mean, std)
    y_test = y_test.to_numpy()

    plt.plot(y_train)
    plt.plot(y_val)
    plt.plot(y_test)
    plt.show()

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
    model.add(InputLayer((5, 1)))
    model.add(LSTM(64, activation='softsign'))
    #model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))

    # model = Sequential()
    # model.add(InputLayer((10, 1)))
    # # model.add(Conv1D(64, kernel_size=2))
    # model.add(LSTM(64))
    # # model.add(Flatten())
    # model.add(Dense(8, 'relu'))
    # model.add(Dense(1, 'linear'))

    cp = keras.callbacks.ModelCheckpoint('model/',
                                         save_best_only=True)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                   metrics=[RootMeanSquaredError()])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[cp])

    y_pred = denormalize(pd.DataFrame(model.predict(X).flatten()), mean, std).to_numpy()
    show_plots(y_pred, y)


if __name__ == '__main__':
    main()
