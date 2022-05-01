# Overview
A CNN-LSTM network for short-term stock market forecasting. The model predicts the closing price value of the following
market day given some time series data containing the following feature set:
- Opening price
- Highest price
- Lowest price
- Closing price
- Volume
- Change (%)

# Usage
Use the following entry point for the program:

`python main.py [-timesteps] [-num_epochs] [-learning_rate]`

# Acknowledgements
Created by Andrew Eldridge, Dylan Neff, and Tori Edmunds for the CSCE 587 final project. The concept for this model was derived from the research of [Lu et al.](https://downloads.hindawi.com/journals/complexity/2020/6622927.pdf)
