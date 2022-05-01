import argparse
import math
import keras.callbacks
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
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


# truncates to n-th decimal place
def truncate(f: float, n: int):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


# display predicted vs true line graph
def plot_predicted_vs_true(y_pred, y_true):
    plt.clf()
    plt.plot(y_pred, label='predicted')
    plt.plot(y_true, label='true')
    plt.legend()
    plt.xlabel('Time step (days)')
    plt.ylabel('Stock value (USD)')
    plt.title('Predicted vs. True Stock Value')
    plt.annotate(
        f'R^2 score = {truncate(r2_score(y_true, y_pred), 4)}',
        (500, 220))
    plt.savefig('predicted_vs_true.png')
    plt.close()


# display parity plot
def plot_parity(y_pred, y_true):
    plt.clf()
    plt.scatter(y_true, y_pred)
    plt.xlabel('True price')
    plt.ylabel('Predicted price')
    plt.title('Predicted vs. True Stock Value')
    plt.annotate(
        f'R^2 score = {truncate(r2_score(y_true, y_pred), 4)}',
        (400, 240))
    plt.savefig('parity.png')
    plt.close()


# column-wise normalization of df
def normalize(data):
    return (data - data.mean()) / data.std()


# de-normalization of df using mean and std from original dataset
def denormalize(data, mean, std):
    return data * std + mean


def main():
    # parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-timesteps', default=10)
    parser.add_argument('-num_epochs', default=100)
    parser.add_argument('-learning_rate', default=0.0001)
    args = parser.parse_args()

    # model params
    timesteps = int(args.timesteps)
    num_features = 6
    num_epochs = int(args.num_epochs)
    learning_rate = float(args.learning_rate)
    input_shape = (None, timesteps, num_features)

    # fetch data
    data = yf.download('SPY', '1980-01-01', '2022-01-01')
    data = data.drop(['Adj Close'], axis=1)
    percent_change = [(data['Close'][i] - data['Close'][i-1]) / data['Close'][i-1] for i in range(1, data.shape[0])]
    data = data[1:]
    data['Change'] = percent_change
    print(data.head(10))

    # split into train/val/test sets
    num_samples = data.shape[0]
    cutoff1 = math.floor(0.7 * num_samples)
    cutoff2 = math.floor(0.9 * num_samples)
    train_data = data[:cutoff1]
    val_data = data[cutoff1:cutoff2]
    test_data = data[cutoff2:]

    # normalize data
    X_train, y_train, _, _ = get_X_y(train_data, timesteps)
    X_val, y_val, _, _ = get_X_y(val_data, timesteps)
    X_test, y_test, test_mean, test_std = get_X_y(test_data, timesteps)

    # build CNN-LSTM network
    model = Sequential()
    model.add(Conv1D(128, 3, activation='relu', input_shape=(timesteps, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(1))
    model.add(LSTM(64, activation='softsign'))
    model.add(Flatten())
    model.add(Dense(1, 'linear'))
    model.build(input_shape=input_shape)
    print(model.summary())

    cp = keras.callbacks.ModelCheckpoint('model/',
                                         save_best_only=True)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=[RootMeanSquaredError()])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, callbacks=[cp])

    # y_pred = denormalize(pd.DataFrame(model.predict(X).flatten()), mean, std).to_numpy()
    mse, rmse = model.evaluate(X_test, y_test)
    print(f'Test loss (MSE): {mse}')
    print(f'Test RMSE: {rmse}')
    y_pred = denormalize(pd.DataFrame(model.predict(X_test).flatten()), test_mean, test_std).to_numpy()
    y_test = denormalize(pd.DataFrame(y_test), test_mean, test_std).to_numpy()
    plot_predicted_vs_true(y_pred, y_test)
    plot_parity(y_pred, y_test)


if __name__ == '__main__':
    main()
