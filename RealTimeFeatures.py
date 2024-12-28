import pandas as pd

class RealTimeFeatures:
    def calculate_sma(self, df, new_row_index):
        """
        Calculate Simple Moving Averages (SMA) for the latest row.
        """
        df.loc[new_row_index, 'sma_10'] = df['close'].iloc[-10:].mean()
        df.loc[new_row_index, 'sma_50'] = df['close'].iloc[-50:].mean()

    def calculate_ema(self, df, new_row_index):
        """
        Calculate Exponential Moving Averages (EMA) for the latest row.
        """
        df.loc[new_row_index, 'ema_10'] = df['close'].ewm(span=10, adjust=False).mean().iloc[-1]

    def calculate_rsi(self, df, new_row_index):
        """
        Calculate Relative Strength Index (RSI) for the latest row.
        """
        delta = df['close'].diff().iloc[-14:]
        gain = delta.where(delta > 0, 0).mean()
        loss = -delta.where(delta < 0, 0).mean()
        rs = gain / loss if loss != 0 else 0
        df.loc[new_row_index, 'rsi_14'] = 100 - (100 / (1 + rs)) if rs != 0 else 100

    def calculate_macd(self, df, new_row_index):
        """
        Calculate Moving Average Convergence Divergence (MACD) for the latest row.
        """
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df.loc[new_row_index, 'macd'] = ema_12.iloc[-1] - ema_26.iloc[-1]
        df.loc[new_row_index, 'macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean().iloc[-1]
        df.loc[new_row_index, 'macd_hist'] = (
            df.loc[new_row_index, 'macd'] - df.loc[new_row_index, 'macd_signal']
        )

    def calculate_bollinger_bands(self, df, new_row_index):
        """
        Calculate Bollinger Bands for the latest row.
        """
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
        std_20 = df['close'].rolling(window=20).std().iloc[-1]
        df.loc[new_row_index, 'bollinger_upper'] = sma_20 + (2 * std_20)
        df.loc[new_row_index, 'bollinger_lower'] = sma_20 - (2 * std_20)

    def calculate_atr(self, df, new_row_index):
        """
        Calculate Average True Range (ATR) for the latest row.
        """
        high_low = df['high'].iloc[-14:] - df['low'].iloc[-14:]
        high_prev_close = (df['high'].iloc[-14:] - df['close'].shift().iloc[-14:]).abs()
        low_prev_close = (df['low'].iloc[-14:] - df['close'].shift().iloc[-14:]).abs()
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        df.loc[new_row_index, 'atr_14'] = true_range.mean()

    def calculate_volume_features(self, df, new_row_index):
        """
        Calculate Volume-Based Features for the latest row.
        """
        volume_ma_20 = df['volume'].rolling(window=20).mean().iloc[-1]
        df.loc[new_row_index, 'volume_ma_20'] = volume_ma_20
        df.loc[new_row_index, 'volume_ratio'] = df.loc[new_row_index, 'volume'] / volume_ma_20 if volume_ma_20 != 0 else 0

    def calculate_roc(self, df, new_row_index):
        """
        Calculate Rate of Change (ROC) for the latest row.
        """
        df.loc[new_row_index, 'roc_10'] = (
            (df['close'].iloc[-1] - df['close'].iloc[-11]) / df['close'].iloc[-11] * 100
            if len(df) >= 11 else 0
        )

    def calculate_lagged_features(self, df, new_row_index):
        """
        Calculate Lagged Features for the latest row.
        """
        df.loc[new_row_index, 'close_lag_1'] = df['close'].iloc[-2] if len(df) >= 2 else None
        df.loc[new_row_index, 'close_lag_3'] = df['close'].iloc[-4] if len(df) >= 4 else None
        df.loc[new_row_index, 'macd_lag_1'] = df['macd'].iloc[-2] if 'macd' in df.columns and len(df) >= 2 else None

    def calculate_candle_features(self, df, new_row_index):
        """
        Calculate Candle Features.
        """
        df.loc[new_row_index, 'candle_body'] = df.loc[new_row_index, 'close'] - df.loc[new_row_index, 'open']
        df.loc[new_row_index, 'upper_wick'] = (
            df.loc[new_row_index, 'high'] - np.maximum(df.loc[new_row_index, 'close'], df.loc[new_row_index, 'open'])
        )
        df.loc[new_row_index, 'lower_wick'] = (
            np.minimum(df.loc[new_row_index, 'close'], df.loc[new_row_index, 'open']) - df.loc[new_row_index, 'low']
        )

