import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from BinanceClient import BinanceClient
# from RealTimeFeatures import RealTimeFeatures
from BatchFeatures import BatchFeatures
import numpy as np
from typing import Final
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

# Initialize Binance client with your API credentials
# dotenv_path = Path('.env-secret')
# load_dotenv(dotenv_path=dotenv_path)
api_secret = os.getenv("BINANCE_SECRET_KEY")
api_key = os.getenv("BINANCE_API_KEY")

# Create Binance client & initialize it
pair = "BTCUSDT"
db_name = pair + "_1min" + "_dry_run.db"
binance_client = BinanceClient(db_name)
binance_client.set_interval("1m")
# realtime_features = RealTimeFeatures()
batch_feature = BatchFeatures()

# Create connection to fetch data
binance_client.make(api_key, api_secret)

# Get current server time
server_time = binance_client.get_server_time()

# Compute start and end time for the last x hours
server_time_dt = datetime.fromtimestamp(server_time['serverTime'] / 1000, tz=datetime.timezone.utc if hasattr(datetime, 'timezone') else None)
end_date = server_time_dt
start_date = server_time_dt - timedelta(hours=10)
start_date_str = int(start_date.timestamp() * 1000)  # Convert to milliseconds
end_date_str = int(end_date.timestamp() * 1000)      # Convert to milliseconds

# Fetch data

df = binance_client.fetch_data(pair, start_date_str, end_date_str)
binance_client.store_data_to_db(pair, df)

# Check if data is fetched
if df.empty:
    print("No data found!!!.")

# The model predicts 'close price' for next 1 hr i.e. 12 candles @ 5_min_interval
span = 10

# create a copy of original df before feature engineering
# Ensure DataFrame has unique indices
# df = df[~df.index.duplicated(keep='last')]

# Feature Engineering (mind the order since some features are depended on others)
# This is initial feature engineering. This will be called again in the loop every time a candle is fetched
batch_feature.calculate_sma(df)
batch_feature.calculate_ema(df)
batch_feature.calculate_rsi(df)
batch_feature.calculate_macd(df)
batch_feature.calculate_bollinger_bands(df)
batch_feature.calculate_atr(df)
batch_feature.calculate_volume_features(df)
batch_feature.calculate_roc(df)
batch_feature.calculate_lagged_features(df)
batch_feature.calculate_candle_features(df)

# Drop NaNs
df.dropna(inplace=True)

# df is sorted by time and indexed by a datetime index
# Drop close from the dataset since 'close' was not used when training the model

# During training, the model's target was defined as 
# df['target'] = (df['close'].shift(12) - df['close']) / df['close'] * 100
# i.e. predict % change in close price after 12 candles (1 hr). 
# So if I want to predict the % change for next candle i.e. next 5 mins then should I drop the last 11 candles 
# of the 'df'? 
X_test = df.drop(['close'], axis=1)

# Load the best model
best_model = joblib.load('best_model_10candles_reduced_1min.joblib')

# Generate predictions
predictions = pd.DataFrame(best_model.predict(X_test)).shift(-span).values  # Predicted percentage changes

# Define buy & sell thresholds. Optimal thresholds were finalized after thorough grid-search 
buy_threshold = -0.13999999999999835
sell_threshold = 0.09

# Main loop starts here
import time

# Initialize a queue to store predictions and their timestamps
prediction_queue = []

import csv
import os

def fetch_and_predict():
    global df
    global prediction_queue

    # Auto-increment log file name
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # Ensure logs directory exists
    base_log_file = os.path.join(log_dir, "trading_log")
    log_file_path = f"{base_log_file}_1.csv"

    # Increment log file name if it already exists
    count = 1
    while os.path.exists(log_file_path):
        count += 1
        log_file_path = f"{base_log_file}_{count}.csv"

    # Write headers to the new log file
    with open(log_file_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(['Timestamp', 'Close', 'Predicted Change', 'Predicted Value', 'Signal',
                         'Trade Action', 'Trade Price', 'Position (BTC)', 'Balance'])

    # Initial balance for testing
    balance = 500  # Starting with $500
    position = 0  # BTC held

    while True:
        new_candle_df = binance_client.fetch_latest_candle(pair)
        if new_candle_df is None or new_candle_df.empty:
            print("Failed to fetch new candle.")
            time.sleep(60)
            continue

        # Add empty columns to align with the main DataFrame
        for column in df.columns:
            if column not in new_candle_df.columns:
                new_candle_df[column] = 0

        # Append to main DataFrame
        df = pd.concat([df, new_candle_df])
        df = df[~df.index.duplicated(keep='last')]  # Ensure unique indices
        df.drop(df.index[0], inplace=True)  # Remove oldest row

        # Compute features for the new candle
        batch_feature.calculate_sma(df)
        batch_feature.calculate_ema(df)
        batch_feature.calculate_rsi(df)
        batch_feature.calculate_macd(df)
        batch_feature.calculate_bollinger_bands(df)
        batch_feature.calculate_atr(df)
        batch_feature.calculate_volume_features(df)
        batch_feature.calculate_roc(df)
        batch_feature.calculate_lagged_features(df)
        batch_feature.calculate_candle_features(df)

        # Generate prediction
        X_new = df.iloc[-1:].drop(['close'], axis=1)
        prediction = best_model.predict(X_new)[0]
        new_close = df.iloc[-1]['close']
        predicted_value_change = new_close * (1 - prediction / 100)

        # Determine signal
        signal = "Sell" if prediction < buy_threshold else "Buy" if prediction > sell_threshold else "Hold"

        # Determine trade action and update balance/position
        trade_action = None
        trade_price = None

        if signal == "Buy" and position == 0:
            trade_action = "Buy"
            trade_price = new_close
            position = balance / new_close  # Calculate BTC bought with $500
            balance = 0  # All funds invested
        elif signal == "Sell" and position > 0:
            trade_action = "Sell"
            trade_price = new_close
            balance = position * new_close  # Convert BTC back to USD
            position = 0  # Exit position

        # Log the signal and trade details
        with open(log_file_path, 'a', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                df.index[-1], new_close, prediction, predicted_value_change, signal,
                trade_action, trade_price, position, balance
            ])

        # Print log to console for monitoring
        print(f"{df.index[-1]} | Close: {new_close:.2f} | Predicted Change: {prediction:.4f} | "
              f"Predicted Value: {predicted_value_change:.2f} | Signal: {signal} | "
              f"Trade Action: {trade_action} | Trade Price: {trade_price} | Position (BTC): {position:.6f} | Balance: {balance:.2f}")

        # Wait for next interval
        time.sleep(60)

# Start the loop
fetch_and_predict()
