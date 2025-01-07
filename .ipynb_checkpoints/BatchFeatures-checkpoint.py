import pandas as pd
import numpy as np

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
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Add lagged EMA features
        for lag in range(1, 11):
            df[f'ema_5_lag_{lag}'] = df['ema_5'].shift(lag)

    def calculate_rsi(self, df, window=14):
        """
        Calculate RSI and smoothed RSI.
        """
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Smoothed RSI
        df['rsi_14_smoothed'] = df['rsi_14'].rolling(window=3).mean()  # Smoothing


    def calculate_macd(self, df):
        """
        Calculate MACD and its smoothed variants.
        """
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Add smoothed MACD histogram
        df['macd_hist_smoothed'] = df['macd_hist'].rolling(window=3).mean()  # 3-period rolling average

        # Add alternative MACD with shorter EMAs
        ema_6 = df['close'].ewm(span=6, adjust=False).mean()
        ema_13 = df['close'].ewm(span=13, adjust=False).mean()
        df['macd_fast'] = ema_6 - ema_13
        df['macd_fast_signal'] = df['macd_fast'].ewm(span=5, adjust=False).mean()
        df['macd_fast_hist'] = df['macd_fast'] - df['macd_fast_signal']


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
        Calculate Rate of Change (ROC) and smoothed ROC variants.
        """
        df['roc_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
        df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
        df['roc_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100

        # Apply smoothing to reduce noise
        df['roc_10_smoothed'] = df['roc_10'].rolling(window=3).mean()  # 3-period rolling average


    def calculate_lagged_features(self, df):
        """
        Calculate Lagged Features.
        """
        df['close_lag_1'] = df['close'].shift(1)
        df['close_lag_3'] = df['close'].shift(3)
        df['close_lag_5'] = df['close'].shift(5)
        df['close_lag_7'] = df['close'].shift(7)
        df['close_lag_9'] = df['close'].shift(9)
        df['close_lag_11'] = df['close'].shift(11)
        df['macd_lag_1'] = df['close'].shift(1)

    def calculate_candle_features(self, df):
        """
        Calculate Candle Features.
        """
        df['candle_body'] = df['close'] - df['open']
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']

    def calculate_stochastic_oscillator(self, df, window=14):
        """
        Calculate the Stochastic Oscillator and its smoothed variants.
        """
        lowest_low = df['low'].rolling(window=window).min()
        highest_high = df['high'].rolling(window=window).max()
        df['stochastic_oscillator'] = ((df['close'] - lowest_low) / (highest_high - lowest_low)) * 100

        # Add a smoothed version (%D)
        df['stochastic_oscillator_slow'] = df['stochastic_oscillator'].rolling(window=3).mean()  # 3-period smoothing


    def calculate_williams_r(self, df, window=14):
        """
        Calculate Williams %R and smoothed Williams %R.
        """
        highest_high = df['high'].rolling(window=window).max()
        lowest_low = df['low'].rolling(window=window).min()

        df['williams_r'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
        df['williams_r_smoothed'] = df['williams_r'].rolling(window=3).mean()  # Smoothing


    def calculate_moving_average_crossover(self, df, short_window=10, long_window=50):
        """
        Calculate Moving Average Crossover.
        """
        df['sma_short'] = df['close'].rolling(window=short_window).mean()
        df['sma_long'] = df['close'].rolling(window=long_window).mean()
        df['ma_crossover'] = df['sma_short'] - df['sma_long']

    def calculate_historical_volatility(self, df, window=14):
        """
        Calculate Historical Volatility.
        """
        df['log_return'] = (df['close'] / df['close'].shift(1)).apply(lambda x: np.log(x))
        df['historical_volatility'] = df['log_return'].rolling(window=window).std()

    def calculate_on_balance_volume(self, df):
        """
        Calculate On-Balance Volume (OBV).
        """
        df['price_change'] = df['close'].diff()
        df['obv'] = (df['price_change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']).cumsum()

    def calculate_money_flow_index(self, df, window=14):
        """
        Calculate Money Flow Index (MFI).
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf_sum = positive_flow.rolling(window=window).sum()
        negative_mf_sum = negative_flow.rolling(window=window).sum()
        mfi = 100 - (100 / (1 + (positive_mf_sum / negative_mf_sum)))
        df['mfi'] = mfi

    def calculate_croc(self, df, window=10):
        """
        Calculate the Cumulative Rate of Change (CROC) to reduce noise in regular ROC.
        """
        roc = (df['close'] - df['close'].shift(window)) / df['close'].shift(window) * 100
        df['croc_10'] = roc.rolling(window=3).mean()  # Smooth ROC further


