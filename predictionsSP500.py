import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- 1. Load data ---
ticker = "^GSPC"
sp500 = yf.Ticker(ticker)
data = sp500.history(period="5y")["Close"]
data = data.dropna()

# --- 2. Train/Test Split ---
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# --- 3. ARIMA ---
arima_model = ARIMA(train, order=(5, 1, 0)).fit()
arima_pred = arima_model.forecast(steps=len(test))
arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))

# --- 4. Exponential Smoothing ---
exp_model = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
exp_pred = exp_model.forecast(len(test))
exp_rmse = np.sqrt(mean_squared_error(test, exp_pred))

# --- 5. Random Forest ---
window = 5  # lagged features
def create_lagged_df(series, window):
    df = pd.DataFrame()
    for i in range(window):
        df[f"lag_{i+1}"] = series.shift(i + 1)
    df["target"] = series.values
    return df.dropna()

rf_data = create_lagged_df(data, window)
train_rf = rf_data[:train_size - window]
test_rf = rf_data[train_size - window:]

X_train_rf, y_train_rf = train_rf.drop("target", axis=1), train_rf["target"]
X_test_rf, y_test_rf = test_rf.drop("target", axis=1), test_rf["target"]

rf_model = RandomForestRegressor()
rf_model.fit(X_train_rf, y_train_rf)
rf_pred = rf_model.predict(X_test_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test_rf, rf_pred))

# --- 6. LSTM ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

X, y = [], []
lookback = 60
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)
X_train_lstm, X_test_lstm = X[:train_size - lookback], X[train_size - lookback:]
y_train_lstm, y_test_lstm = y[:train_size - lookback], y[train_size - lookback:]

lstm_model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(X_train_lstm.shape[1], 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=0)

lstm_pred = lstm_model.predict(X_test_lstm)
lstm_pred_rescaled = scaler.inverse_transform(lstm_pred)
y_test_lstm_rescaled = scaler.inverse_transform(y_test_lstm)

lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm_rescaled, lstm_pred_rescaled))

# --- 7. Results ---
print(f"ARIMA RMSE: {arima_rmse:.2f}")
print(f"Exponential Smoothing RMSE: {exp_rmse:.2f}")
print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"LSTM RMSE: {lstm_rmse:.2f}")

# --- 8. Plot (Optional) ---
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label="Actual", color="black")
plt.plot(test.index, arima_pred, label="ARIMA")
plt.plot(test.index, exp_pred, label="Exponential Smoothing")
plt.plot(test_rf.index, rf_pred, label="Random Forest")
plt.plot(test.index[-len(lstm_pred_rescaled):], lstm_pred_rescaled.flatten(), label="LSTM")
plt.title("S&P 500 Forecast Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

