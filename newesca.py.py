#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import yfinance as yfin
from scipy.stats import norm

# Load and preprocess data
def load_data(symbol, start_date, end_date):
    st.write(f"Fetching {symbol} data from {start_date} to {end_date}...")
    data = yfin.download(symbol, start=start_date, end=end_date)
    data = data[['Close']]  # Use only the closing prices
    data.index.name = 'Date'  # Ensure index is named 'Date'
    return data

def preprocess_data(data, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # LSTM input shape
    return X, y, scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(model, last_data, scaler, window_size=60, steps=30):
    predictions = []
    last_window = last_data[-window_size:]
    scaled_last_window = scaler.transform(last_window)

    for _ in range(steps):
        scaled_last_window = np.reshape(scaled_last_window, (1, window_size, 1))
        pred = model.predict(scaled_last_window)[0][0]
        predictions.append(pred)

        scaled_last_window = np.append(scaled_last_window[0][1:], pred).reshape(window_size, 1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

def calculate_prediction_probability(predictions, actual, std_dev):
    probabilities = []
    for pred in predictions:
        prob = norm.cdf((pred - actual) / std_dev) * 100
        probabilities.append(prob)
    return probabilities

# Streamlit Dashboard
def main():
    st.title("EUR/USD Exchange Rate Prediction")
    st.markdown("This dashboard uses an optimized LSTM model to predict EUR/USD exchange rates and provide future price trends.")

    # Symbol selection and date range input
    symbol = st.text_input("Enter the forex symbol (e.g., EURUSD=X):", value="EURUSD=X")
    start_date = st.date_input("Select Start Date", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("Select End Date", value=pd.to_datetime("2020-12-31"))

    if st.button("Fetch Data"):
        data = load_data(symbol, start_date, end_date)
        st.write("Data preview:", data.head())

        # Train-Test Split
        window_size = 60
        train_data = data.iloc[:-100]
        test_data = data.iloc[-100:]

        # Preprocess data
        X_train, y_train, scaler = preprocess_data(train_data[['Close']], window_size)
        X_test, y_test, _ = preprocess_data(test_data[['Close']], window_size)

        # Build and Train Model
        model = build_model((X_train.shape[1], 1))
        with st.spinner("Training the model..."):
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        st.success("Model trained successfully!")

        # Evaluate Model
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate Metrics
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))

        st.subheader("Model Performance Metrics")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")

        # Predict Future Prices
        future_prices = predict_future_prices(model, test_data[['Close']], scaler, window_size, steps=30)

        # Calculate probabilities
        std_dev = np.std(actual - predictions)
        probabilities = calculate_prediction_probability(future_prices, actual[-1][0], std_dev)

        # Display Predictions
        st.subheader("Future Predictions")
        next_day_price = future_prices[0][0]
        next_week_prices = future_prices[:7].flatten()
        next_month_prices = future_prices.flatten()

        st.write(f"**Next-Day Prediction:** {next_day_price:.4f} (Probability: {probabilities[0]:.2f}%)")
        st.write(f"**Next Week Predictions:** {next_week_prices} (Average Probability: {np.mean(probabilities[:7]):.2f}%)")
        st.write(f"**Next Month Predictions:** {next_month_prices} (Average Probability: {np.mean(probabilities):.2f}%)")

        # Plot Future Predictions
        st.subheader("Future Price Trend")
        future_dates = pd.date_range(test_data.index[-1] + pd.Timedelta(days=1), periods=30)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(future_dates, future_prices, label="Future Prices", color="green")
        ax.set_title("Future Price Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()

