# Import the class from the Python file (module)
from BinanceClient import BinanceClient
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from pathlib import Path

fetch_data_from_binance = False

# Initialize Binance client with your API credentials
dotenv_path = Path('.env-secret')
load_dotenv(dotenv_path=dotenv_path)
api_secret = os.getenv("BINANCE_SECRET_KEY")
api_key = os.getenv("BINANCE_API_KEY")

binance_client = BinanceClient(api_key, api_secret)

# Define trading pair and date range
pair = "BTCUSDT"
start_date = "01 Jan, 2024"
end_date = "01 Dec, 2024"

if fetch_data_from_binance:
    # Fetch data
    data = binance_client.fetch_data(pair, start_date, end_date)
    binance_client.store_data_to_db(pair, data)
else:
    data = binance_client.fetch_data_from_db(pair, start_date, end_date)
        
# Check if data is fetched
if not data.empty:
    # Convert the fetched data into a pandas DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Set the timestamp as the index
    df.set_index('timestamp', inplace=True)
    
    # Convert the columns to numeric (some values may be strings by default)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)

    # Plot the closing price
    df['close'].plot(figsize=(10, 6), title=f'{pair} Close Price', xlabel='Date', ylabel='Price (USDT)')
    plt.show() 
else:
    print("No data fetched.")