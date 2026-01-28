import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path
from BinanceClient import BinanceClient
from BatchFeatures import BatchFeatures
import joblib
from datetime import datetime, timedelta
import time
import csv

# Initialize Binance client with your API credentials
# dotenv_path = Path('.env-secret')
# load_dotenv(dotenv_path=dotenv_path)
api_secret = os.getenv("BINANCE_SECRET_KEY")
api_key = os.getenv("BINANCE_API_KEY")

# Create Binance client & initialize it
pair = "BTCUSDT"
db_name = pair + "_1min_ema5_n10" + "_dry_run.db"
binance_client = BinanceClient(db_name)
binance_client.set_interval("1m")
bf = BatchFeatures()

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

# Feature Engineering (mind the order since some features are depended on others)
# This is initial feature engineering. This will be called again in the loop every time a candle is fetched
# Must-have features

# Feature Engineering (mind the order since some features are depended on others)
bf = BatchFeatures()
bf.calculate_sma(df)
bf.calculate_ema(df)
bf.calculate_rsi(df)
bf.calculate_macd(df)
bf.calculate_bollinger_bands(df)
bf.calculate_atr(df)
bf.calculate_volume_features(df)
bf.calculate_roc(df)
bf.calculate_lagged_features(df)
bf.calculate_candle_features(df)
bf.calculate_stochastic_oscillator(df)
bf.calculate_williams_r(df)
bf.calculate_moving_average_crossover(df)
bf.calculate_historical_volatility(df)
bf.calculate_on_balance_volume(df)
bf.calculate_money_flow_index(df)
bf.calculate_croc(df)

# Drop NaNs
df.dropna(inplace=True)

df_raw = df.copy()

# Load the best model
best_model = joblib.load('lstm_n10candles_1min_ema5.joblib')

# Load the scaler and LSTM model
scaler = joblib.load("lstm_scaler_em5_n10.pkl")  # Ensure you saved your scaler during training

# Define buy & sell thresholds. Optimal thresholds were finalized after thorough grid-search 
sell_threshold = 0.01
buy_threshold =  -0.0142695

def initialize_log_file(log_dir="logs", base_log_file="trading_log"):
    """
    Initialize and increment log file name if it already exists.
    """
    os.makedirs(log_dir, exist_ok=True)
    count = 1
    log_file_path = os.path.join(log_dir, f"{base_log_file}_{count}.csv")
    
    while os.path.exists(log_file_path):
        count += 1
        log_file_path = os.path.join(log_dir, f"{base_log_file}_{count}.csv")

    with open(log_file_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "Timestamp",
                "Close",
                "Predicted Change",
                "Predicted Value",
                "EMA_5",
                "EMA_10",
                "Target",
                "Signal",
                "Trade Action",
                "Trade Price",
                "Position (BTC)",
                "Balance",
            ]
        )
    return log_file_path

def fetch_latest_data():
    """
    Fetch the latest candle data and append it to the main DataFrame.
    """
    global df
    global df_raw
 
    new_candle_df = binance_client.fetch_latest_candle(pair)
    if new_candle_df is None or new_candle_df.empty:
        print("Failed to fetch new candle.")
        time.sleep(60)
        return None

    # update the raw_df
    df_raw = pd.concat([df_raw, new_candle_df])
    df_raw.drop(df_raw.index[0], inplace=True)  # Remove the oldest row

    # make a copy and work on that. keep original df safe
    df = df_raw.copy()
    df = df[~df.index.duplicated(keep="last")]  # Ensure unique indices

    return df

def compute_features():
    """
    Compute all required features for the dataset.
    """
    # Feature Engineering (mind the order since some features are depended on others)
    bf = BatchFeatures()
    bf.calculate_sma(df)
    bf.calculate_ema(df)
    bf.calculate_rsi(df)
    bf.calculate_macd(df)
    bf.calculate_bollinger_bands(df)
    bf.calculate_atr(df)
    bf.calculate_volume_features(df)
    bf.calculate_roc(df)
    bf.calculate_lagged_features(df)
    bf.calculate_candle_features(df)
    bf.calculate_stochastic_oscillator(df)
    bf.calculate_williams_r(df)
    bf.calculate_moving_average_crossover(df)
    bf.calculate_historical_volatility(df)
    bf.calculate_on_balance_volume(df)
    bf.calculate_money_flow_index(df)
    bf.calculate_croc(df)
    
    df.dropna(inplace=True)

def calculate_target(nn=10):
    """
    Calculate the target for the dataset.
    """
    # Predict the percentage price change over the next 'n' candles
    df['target'] = (df['ema_5'] - df['ema_5'].shift(-nn)) / df['ema_5'] * 100

    for ii in range(1, 2 * nn):
        df[f'target_lag_{ii}'] = df['target'].shift(ii)

    df.dropna(inplace=True)

def generate_prediction(seq_length):
    """
    Prepare input for LSTM and generate a prediction.
    """
    # Extract the last `seq_length` rows, dropping the 'target' column
    X_new = df.iloc[-seq_length:].drop(columns=['target']).to_numpy()

    # Ensure X_new has the correct number of rows
    if X_new.shape[0] != seq_length:
        print(f"Warning: X_new has {X_new.shape[0]} rows instead of {seq_length}. Adjusting...")
        X_new = X_new[-seq_length:]  # Take the last `seq_length` rows

    # Ensure X_new has valid feature names for the scaler
    X_new_df = pd.DataFrame(X_new, columns=scaler.feature_names_in_)

    # Apply scaling
    X_new_scaled = scaler.transform(X_new_df)

    # Reshape for LSTM
    try:
        X_new_scaled = X_new_scaled.reshape(1, seq_length, -1)
    except ValueError as e:
        print(f"Error reshaping X_new_scaled: {e}")
        print(f"X_new_scaled shape: {X_new_scaled.shape}")
        raise

    # Generate prediction
    prediction = best_model.predict(X_new_scaled)[0, 0]  # Extract scalar prediction
    return prediction


def determine_signal(prediction):
    """
    Determine the signal based on the prediction.
    """
    predicted_value_change = df.iloc[-1]['ema_5'] * (1 + prediction/100)
    signal = (
        "Sell" if prediction > sell_threshold else
        "Buy" if prediction < buy_threshold else
        "Hold"
    )
    return signal, predicted_value_change

def execute_trade(signal, new_close, position, balance):
    """
    Execute the trade and update balance/position.
    """
    trade_action = None
    trade_price = None

    if signal == "Buy" and position == 0:
        trade_action = "Buy"
        trade_price = new_close
        position = balance / new_close  # Calculate BTC bought
        balance = 0  # All funds invested
    elif signal == "Sell" and position > 0:
        trade_action = "Sell"
        trade_price = new_close
        balance = position * new_close  # Convert BTC back to USD
        position = 0  # Exit position

    return trade_action, trade_price, position, balance

def fetch_and_predict():
    """
    Fetch, process, and predict in a loop.
    """
    global df

    log_file_path = initialize_log_file()
    balance = 500  # Initial balance
    position = 0  # BTC held
    seq_length = 60  # Sequence length used during training

    while True:
        df = fetch_latest_data()
        if df is None:
            continue

        compute_features()
        calculate_target(5)
        prediction = generate_prediction(seq_length)

        new_close = df.iloc[-1]["close"]
        signal, predicted_value_change = determine_signal(prediction)
        trade_action, trade_price, position, balance = execute_trade(signal, new_close, position, balance)

        # Log the results
        # Log the results
        with open(log_file_path, "a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(
                [
                    df.index[-1],
                    new_close,
                    prediction,
                    predicted_value_change,
                    df.iloc[-1]['ema_5'],
                    df.iloc[-1]['ema_10'],
                    df.iloc[-1]['target'],
                    signal,
                    trade_action,
                    trade_price,
                    position,
                    balance,
                ]
            )

        # Print log to console for monitoring
        print(
            f"{df.index[-1]} | Close: {new_close:.2f} | Predicted Change: {prediction:.4f} | "
            f"Predicted Value: {predicted_value_change:.2f} | EMA_5: {df.iloc[-1]['ema_5']:.2f} | "
            f"EMA_10: {df.iloc[-1]['ema_10']:.2f} | Target: {df.iloc[-1]['target']:.4f} | Signal: {signal} | "
            f"Trade Action: {trade_action} | Trade Price: {trade_price} | Position (BTC): {position:.6f} | Balance: {balance:.2f}"
        )


        # Wait for the next interval
        time.sleep(60)
           

# Start the loop
fetch_and_predict()
