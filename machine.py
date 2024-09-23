import subprocess
import talib as ta
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import plotly.graph_objs as go
from datetime import datetime
from colorama import Fore

ticker = "AAPL"
stock_data = yf.download(ticker,start=datetime(2019,1,1),end=datetime.now())

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.65)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:len(scaled_data)]

# Function to create sequences for LSTM input
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 100  # Length of input sequences
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Reshape the input data to match the expected shape for LSTM
X_train = X_train.reshape(-1, seq_length, 1)
X_test = X_test.reshape(-1, seq_length, 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Take the last sequence from the test data to start forecasting
last_sequence = X_test[-1]
future_steps = 30
# Make predictions for future time steps
forecast = []
for _ in range(future_steps):
    # Reshape the last sequence to match the input shape
    last_sequence_reshaped = last_sequence.reshape(1, seq_length, 1)
    # Predict the next value
    next_prediction = model.predict(last_sequence_reshaped)
    # Append the prediction to the forecast list
    forecast.append(next_prediction[0][0])  # Extracting the scalar value from the prediction
    # Update the last sequence by removing the first element and adding the prediction
    last_sequence = np.append(last_sequence[1:], next_prediction)

# Inverse transform the forecasted values to get the actual price
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# Prepare data for plotting
dates = stock_data.index[-100:].strftime('%Y-%m-%d')
close_prices = stock_data['Close'].values[-100:]
forecast_dates = pd.date_range(start=stock_data.index[-1], periods=future_steps + 1).strftime('%Y-%m-%d')
forecast_prices = np.concatenate((stock_data['Close'].values[-1:], forecast.flatten()), axis=None)

# Create interactive plot
fig = go.Figure()
# Plot last 100 days close price
fig.add_trace(go.Scatter(x=dates, y=close_prices, mode='lines', name='Historical Close Price'))
# Plot forecasted prices
fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices, mode='lines', name='Forecast Price'))
# Update layout for better readability
fig.update_layout(title=f"Stock Price Forecast for {ticker}",
                  xaxis_title="Date",
                  yaxis_title="Price",
                  xaxis_rangeslider_visible=True)
# Display the plot using Streamlit
st.plotly_chart(fig)

from sklearn.metrics import mean_squared_error

# Calculate MSE for training and testing data
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

# Calculate RMSE for training and testing data
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

# Print MSE and RMSE for training and testing data
print('Train MSE:', train_mse)
print('Test MSE:', test_mse)
print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)

# Extract actual values for comparison with forecast
forecast_actual = stock_data['Close'].values[-len(forecast):]

# Calculate MSE for forecasting
forecast_mse = mean_squared_error(forecast_actual, forecast)
forecast_rmse = np.sqrt(forecast_mse)

# Print MSE and RMSE for forecasting
print('Forecast MSE:', forecast_mse)
print('Forecast RMSE:', forecast_rmse)

# Calculate Mean Absolute Percentage Error (MAPE) for training and testing data
train_mape = np.mean(np.abs((y_train - train_predictions) / y_train)) * 100
test_mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100

# Print MAPE for training and testing data
print('Train MAPE:', train_mape)
print('Test MAPE:', test_mape)

# Calculate Mean Absolute Percentage Error (MAPE) for forecasting
forecast_actual = stock_data['Close'].values[-1]  # Actual value for the last known day
forecast_mape = np.mean(np.abs((forecast_actual - forecast.flatten()) / forecast_actual)) * 100

# Print MAPE for forecasting
print('Forecast MAPE:', forecast_mape)

print('Train Loss:', train_loss)
print('Test Loss:', test_loss)

