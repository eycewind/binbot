import sqlite3
from binance.client import Client
import pandas as pd
import time
import datetime

class BinanceClient:
    def __init__(self, db_name="binance_data.db"):
        """
        Initialize the BinanceClient database name.
        """
        self.db_name = db_name
        self.create_db()

    def make(self, api_key, api_secret):
        """
        Initialize the BinanceClient with API credentials and database name.
        """
        self.client = Client(api_key, api_secret)
 
    def set_interval(self, interval):
        """
        Set the interval for BinanceClient.
        
        :param interval: Desired interval as a string (e.g., "1m", "5m", "1h").
        """
        valid_intervals = {
            "1s": Client.KLINE_INTERVAL_1SECOND,
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "3m": Client.KLINE_INTERVAL_3MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "30m": Client.KLINE_INTERVAL_30MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "2h": Client.KLINE_INTERVAL_2HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "6h": Client.KLINE_INTERVAL_6HOUR,
            "8h": Client.KLINE_INTERVAL_8HOUR,
            "12h": Client.KLINE_INTERVAL_12HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
            "3d": Client.KLINE_INTERVAL_3DAY,
            "1w": Client.KLINE_INTERVAL_1WEEK,
            "1M": Client.KLINE_INTERVAL_1MONTH
        }
        
        if interval in valid_intervals:
            self.interval = valid_intervals[interval]
            return self.interval
        else:
            raise ValueError(f"Invalid interval '{interval}'. Valid options are: {', '.join(valid_intervals.keys())}")

    def create_db(self):
        """
        Create the database and the table if they don't already exist.
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS candlesticks (
                                pair TEXT,
                                timestamp INTEGER,
                                open REAL,
                                high REAL,
                                low REAL,
                                close REAL,
                                volume REAL,
                                PRIMARY KEY (pair, timestamp))''')
            conn.commit()

    def store_data_to_db(self, pair, data):
        """
        Store fetched candlestick data to the database.

        :param pair: The trading pair, e.g., 'BTCUSDT'.
        :param data: A list of candlestick data fetched from Binance.
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            # Iterate over the rows of the DataFrame
            for index, row in data.iterrows():
                try:
                    timestamp = int(index.timestamp() * 1000)  # Convert index (timestamp) to milliseconds
                    open_price = float(row['open'])
                    high = float(row['high'])
                    low = float(row['low'])
                    close = float(row['close'])
                    volume = float(row['volume'])
                    cursor.execute('''INSERT OR REPLACE INTO candlesticks
                                    (pair, timestamp, open, high, low, close, volume)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                                (pair, timestamp, open_price, high, low, close, volume))
                except (ValueError, KeyError) as e:
                    print(f"Skipping invalid row: {row}. Error: {e}")
            conn.commit()

    # Convert start_date and end_date to Unix timestamps in milliseconds
    def convert_to_timestamp_in_ms(self, date_str):
        dt = datetime.datetime.strptime(date_str, '%d %b, %Y')
        timestamp_in_seconds = int(time.mktime(dt.timetuple()))  # Unix timestamp in seconds
        return timestamp_in_seconds * 1000  # Convert to milliseconds

    def fetch_data_from_db(self, pair):
        """
        Fetch all candlestick data from the database for a specific pair.

        :param pair: The trading pair, e.g., 'BTCUSDT'.
        :return: A pandas DataFrame of historical candlesticks data or None if not found.
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT * FROM candlesticks WHERE pair = ?''', (pair,))
            rows = cursor.fetchall()

            # cursor.close()
            # conn.close()

            if not rows:
                return None

            # Create DataFrame from the fetched data
            df = pd.DataFrame(rows, columns=['pair', 'timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # pair infor is not present in df when its fetched from Binance so its dropped here
            df.drop(['pair'], inplace=True, axis=1)

            # Ensure all numeric columns are properly typed
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            

            return df


    def fetch_data(self, pair, start_date, end_date):
        """
        Fetch historical candlestick data from Binance.
        
        :param pair: The trading pair, e.g., 'BTCUSDT'.
        :param start_date: Start date in format 'DD MMM, YYYY'.
        :param end_date: End date in format 'DD MMM, YYYY'.
        :param interval: The interval for the candlesticks, default is 5 minutes.
        :return: A list of historical candlesticks data or None if there is an error.
        """
        # Check if data is already available in the database
        # existing_data = self.fetch_data_from_db(pair, start_date, end_date)
        # if existing_data is not None:
            # print("Fetching data from database...")
            # return existing_data
        
        # If not available, fetch data from Binance API
        try:
            candles = self.client.get_historical_klines(pair, self.interval, start_date, end_date)
            # Store the data in the database for future use
            # self.store_data_to_db(pair, candles)
            print("Fetching data from Binance API...")
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_value', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asse_volume', 'Ignore'])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.apply(pd.to_numeric)
            return df
        except Exception as e:
            print(f"Error fetching data from Binance: {e}")
            return None
        
    def get_server_time(self):
        return self.client.get_server_time()

    def fetch_latest_candle(self, pair):
        """
        Fetch the most recent candle for the given trading pair.
        
        :param pair: The trading pair, e.g., 'BTCUSDT'.
        :return: The latest candle data as a DataFrame.
        """
        try:
            candles = self.client.get_klines(symbol=pair, interval=self.interval, limit=1)
            if candles:
                df = pd.DataFrame(candles, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_value', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df.apply(pd.to_numeric)
                return df
            else:
                print("No candles fetched.")
                return pd.DataFrame()  # Return an empty DataFrame for consistency
        except Exception as e:
            print(f"Error fetching latest candle: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of error
