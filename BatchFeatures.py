import pandas as pd

class BatchFeatures:
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
