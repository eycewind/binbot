# Import the class from the Python file (module)
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from BinanceClient import BinanceClient
import numpy as np
from typing import Final
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Initialize Binance client with your API credentials
dotenv_path = Path('.env-secret')
load_dotenv(dotenv_path=dotenv_path)
api_secret = os.getenv("BINANCE_SECRET_KEY")
api_key = os.getenv("BINANCE_API_KEY")

# Create Binance client & initialize it
binance_client = BinanceClient()

fetch_data_from_binance = False
# Define trading pair and date range
pair = "BTCUSDT"
start_date = "01 Jan, 2024"
end_date = "03 Jan, 2024"


if fetch_data_from_binance:
    # Create connection to fecth data
    binance_client.make(api_key, api_secret)

    # Fetch data
    data = binance_client.fetch_data(pair, start_date, end_date)
    binance_client.store_data_to_db(pair, data)
else:
    data = binance_client.fetch_data_from_db(pair, start_date, end_date)

 # Check if data is fetched
if not data.empty:
    # Convert the fetched data into a pandas DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Set the timestamp as the index
    df.set_index('timestamp', inplace=True)
    
    # Convert the columns to numeric (some values may be strings by default)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
else:
    print("No data found!!!.")  

# Define target
# Predict the percentage price change over the next 'n' candles
nn = 12         # predict change over 1 hr for 5MIN_INTERVAL
df['target'] = (df['close'].shift(nn) - df['close']) / df['close'] * 100

# Feature Engineering (mind the order since some features are depended on others)
binance_client.calculate_sma(df)
binance_client.calculate_ema(df)
binance_client.calculate_rsi(df)
binance_client.calculate_macd(df)
binance_client.calculate_bollinger_bands(df)
binance_client.calculate_atr(df)
binance_client.calculate_volume_features(df)
binance_client.calculate_roc(df)
binance_client.calculate_lagged_features(df)
binance_client.calculate_candle_features(df)


## Test/train split
# drop NaNs
df.dropna(inplace=True)

# df is sorted by time and indexed by a datetime index
n = len(df)
train_end = int(n * 0.8) # 80% data used for training
train_df = df.iloc[:train_end]
test_df = df.iloc[train_end:]

# Separate features and target
X_train = train_df.drop(['target', 'close'], axis=1)
y_train = train_df['target']

X_test = test_df.drop(['target', 'close'], axis=1)
y_test = test_df['target']

# Load the best model
best_model = joblib.load('best_model_12candles.joblib')

# Generate predictions
predictions = pd.DataFrame(best_model.predict(X_test))  # Predicted percentage changes

# Define buy & sell thrsholds
buy_threshold = -0.37999999999999856
sell_threshold = 0.29000000000000004

# Generate trade signals
# Generate signals with reversed logic
trading_signals = [
    "Buy" if pred < buy_threshold else "Sell" if pred > sell_threshold else "Hold"
    for pred in results_df['Predicted Change']
]





