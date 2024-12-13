import sqlite3
from binance.client import Client
import pandas as pd
import time
import datetime

class BinanceClient:
    def __init__(self, api_key, api_secret, db_name="binance_data.db"):
        """
        Initialize the BinanceClient with API credentials and database name.
        """
        self.client = Client(api_key, api_secret)
        self.db_name = db_name
        self.create_db()

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
            for candle in data:
                timestamp = candle[0]
                open_price = float(candle[1])
                high = float(candle[2])
                low = float(candle[3])
                close = float(candle[4])
                volume = float(candle[5])
                cursor.execute('''INSERT OR REPLACE INTO candlesticks
                                  (pair, timestamp, open, high, low, close, volume)
                                  VALUES (?, ?, ?, ?, ?, ?, ?)''',
                               (pair, timestamp, open_price, high, low, close, volume))
            conn.commit()

    # Convert start_date and end_date to Unix timestamps in milliseconds
    def convert_to_timestamp_in_ms(self, date_str):
        dt = datetime.datetime.strptime(date_str, '%d %b, %Y')
        timestamp_in_seconds = int(time.mktime(dt.timetuple()))  # Unix timestamp in seconds
        return timestamp_in_seconds * 1000  # Convert to milliseconds

    def fetch_data_from_db(self, pair, start_date, end_date):
        """
        Fetch candlestick data from the database.
        
        :param pair: The trading pair, e.g., 'BTCUSDT'.
        :param start_date: Start date in format 'DD MMM, YYYY'.
        :param end_date: End date in format 'DD MMM, YYYY'.
        :return: A pandas DataFrame of historical candlesticks data or None if not found.
        """
        start_date_unix_ms = self.convert_to_timestamp_in_ms(start_date)
        end_date_unix_ms = self.convert_to_timestamp_in_ms(end_date)

        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT * FROM candlesticks WHERE pair = ? AND timestamp BETWEEN ? AND ?''',
                           (pair, start_date_unix_ms, end_date_unix_ms))
            rows = cursor.fetchall()
            
            cursor.close()
            if not rows:
                return None
            
            df = pd.DataFrame(rows, columns=['pair', 'timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df

    def fetch_data(self, pair, start_date, end_date, interval=Client.KLINE_INTERVAL_5MINUTE):
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
            candles = self.client.get_historical_klines(pair, interval, start_date, end_date)
            # Store the data in the database for future use
            self.store_data_to_db(pair, candles)
            print("Fetching data from Binance API...")
            return candles
        except Exception as e:
            print(f"Error fetching data from Binance: {e}")
            return None
