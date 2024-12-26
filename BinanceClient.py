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
            # self.store_data_to_db(pair, candles)
            print("Fetching data from Binance API...")
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_value', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asse_volume', 'Ignore'])
            df.drop(['timestamp', 'open', 'high', 'low', 'close', 'volume'], axis=1)
            return df
        except Exception as e:
            print(f"Error fetching data from Binance: {e}")
            return None

    def calculate_sma(self, df):
        """
        Calculate Simple Moving Averages (SMA).
        """
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

    def calculate_ema(self, df):
        """
        Calculate Exponential Moving Averages (EMA).
        """
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

    def calculate_rsi(self, df):
        """
        Calculate Relative Strength Index (RSI).
        """
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

    def calculate_macd(self, df):
        """
        Calculate Moving Average Convergence Divergence (MACD).
        """
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

    def calculate_bollinger_bands(self, df):
        """
        Calculate Bollinger Bands.
        """
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = sma_20 + (2 * std_20)
        df['bollinger_lower'] = sma_20 - (2 * std_20)

    def calculate_atr(self, df):
        """
        Calculate Average True Range (ATR).
        """
        high_low = df['high'] - df['low']
        high_prev_close = (df['high'] - df['close'].shift()).abs()
        low_prev_close = (df['low'] - df['close'].shift()).abs()
        true_range = high_low.to_frame('hl').join(high_prev_close.to_frame('hpc')).join(low_prev_close.to_frame('lpc')).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()

    def calculate_volume_features(self, df):
        """
        Calculate Volume-Based Features.
        """
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']

    def calculate_roc(self, df):
        """
        Calculate Rate of Change (ROC).
        """
        df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100

    def calculate_lagged_features(self, df):
        """
        Calculate Lagged Features.
        """
        df['close_lag_1'] = df['close'].shift(1)
        df['close_lag_3'] = df['close'].shift(3)
        df['macd_lag_1'] = df['close'].shift(1)

    def calculate_candle_features(self, df):
        """
        Calculate Candle Features.
        """
        df['candle_body'] = df['close'] - df['open']
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
