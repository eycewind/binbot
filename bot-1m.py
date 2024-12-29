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
db_name = pair + "_1min" + "_dry_run"
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

def fetch_and_predict():
    global df
    global prediction_queue
    while True:
        new_candle_df = binance_client.fetch_latest_candle(pair)
        if new_candle_df is None or new_candle_df.empty:
            print("Failed to fetch new candle.")
            time.sleep(60)
            continue

        # Add the new data to the database
        # binance_client.store_data_to_db(pair, new_candle_df.reset_index().values.tolist())

        # Add empty columns to align with main DataFrame
        for column in df.columns:
            if column not in new_candle_df.columns:
                new_candle_df[column] = 0

        # Append to main DataFrame
        df = pd.concat([df, new_candle_df])
        # print("Appended new candle to DataFrame.")

        # Ensure unique indices
        df = df[~df.index.duplicated(keep='last')]

        # Remove the oldest row to prevent infinite growth
        df.drop(df.index[0], inplace=True)
        # print("Dropped the oldest entry to prevent infinite growth.")

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

        signal = "Buy" if prediction < buy_threshold else "Sell" if prediction > sell_threshold else "Hold"

        # Log the signal
        log_message = (
            f"{df.index[-1]}, Close: {new_close:.2f}, "
            f"PredictedChange:, {prediction:.4f}, PredictedValue:, {predicted_value_change:.2f}, Signal: {signal} "
        )

        print(log_message)
        with open('trading_log.txt', 'a') as log_file:
            log_file.write(log_message + '\n')

        # Wait for next interval
        # print("Waiting for the next 1-minute interval...")
        time.sleep(60)

# Start the loop
fetch_and_predict()
