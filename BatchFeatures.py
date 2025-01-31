import pandas as pd
import numpy as np

class BatchFeatures:
    def calculate_ema(self, df, spans=[5, 10, 50]):
        """
        Calculate Exponential Moving Averages (EMA).
        """
        for span in spans:
            df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

    def calculate_rsi(self, df, windows=[14]):
        """
        Calculate RSI for multiple window sizes and smoothed RSI variants.

        Args:
            df (pd.DataFrame): Input DataFrame with a 'close' column.
            windows (list): List of window sizes for calculating RSI.
        """
        for window in windows:
            # Calculate price changes
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Calculate average gains and losses
            avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

            # Calculate RSI
            # add epsilon to prevent division by zero
            rs = avg_gain / (avg_loss + 1e-10)
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))

            # Smoothed RSI
            df[f'rsi_{window}_smoothed'] = df[f'rsi_{window}'].rolling(window=3).mean()  # 3-period smoothing


    def calculate_macd(self, df, spans={'standard': (12, 26, 9), 'fast': (6, 13, 5)}):
        """
        Calculate MACD and its smoothed variants.

        Args:
            df (pd.DataFrame): Input DataFrame with a 'close' column.
            spans (dict): Dictionary specifying spans for 'standard' and 'fast' MACD.
                        Format: {'standard': (fast_span, slow_span, signal_span),
                                'fast': (fast_span, slow_span, signal_span)}
        """
        if 'ema_12' not in df or 'ema_26' not in df:
            self.calculate_ema(df, spans=[12, 26])
        # Standard MACD
        fast_span, slow_span, signal_span = spans['standard']
        ema_fast = df['close'].ewm(span=fast_span, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_span, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal_span, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Smoothed MACD Histogram (Rolling Average)
        df['macd_hist_smoothed'] = df['macd_hist'].rolling(window=3).mean()

        # Fast MACD (Alternative)
        fast_span_fast, slow_span_fast, signal_span_fast = spans['fast']
        ema_fast_fast = df['close'].ewm(span=fast_span_fast, adjust=False).mean()
        ema_slow_fast = df['close'].ewm(span=slow_span_fast, adjust=False).mean()
        df['macd_fast'] = ema_fast_fast - ema_slow_fast
        df['macd_fast_signal'] = df['macd_fast'].ewm(span=signal_span_fast, adjust=False).mean()
        df['macd_fast_hist'] = df['macd_fast'] - df['macd_fast_signal']


    def calculate_bollinger_bands(self, df, window=20, num_std_dev=2):
        """
        Calculate Bollinger Bands for given window size and standard deviation multiplier.

        Args:
            df (pd.DataFrame): Input DataFrame with a 'close' column.
            window (int): Rolling window size for the SMA and standard deviation.
            num_std_dev (float): Multiplier for the standard deviation.
        """
        # Calculate rolling mean (SMA) and standard deviation
        sma = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()

        # Calculate Bollinger Bands
        df[f'bollinger_upper_{window}'] = sma + (num_std_dev * std)
        df[f'bollinger_lower_{window}'] = sma - (num_std_dev * std)
        df[f'bollinger_middle_{window}'] = sma  # Optional: Middle band (SMA)

    def calculate_volume_features(self, df, windows=[20]):
        """
        Calculate Volume-Based Features for multiple window sizes.

        Args:
            df (pd.DataFrame): Input DataFrame with a 'volume' column.
            windows (list): List of rolling window sizes for calculating volume features.
        """
        for window in windows:
            # Calculate moving average of volume
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
            
            # Preserve backward compatibility for volume_ratio with the default 20-window
            if window == 20:
                df['volume_ratio'] = df['volume'] / df[f'volume_ma_{window}']
            else:
                # For other windows, use window-specific column names
                df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']

    def calculate_candle_features(self, df, legacy_compatibility=True):
        """
        Calculate Candle Features with an optional 'candle_range' feature.

        Args:
            df (pd.DataFrame): Input DataFrame with 'open', 'close', 'high', and 'low' columns.
            legacy_compatibility (bool): If True, only calculates 'candle_body', 'upper_wick', and 'lower_wick'.
                                        If False, also adds 'candle_range'.
        """
        # Core candle features (always calculated)
        df['candle_body'] = df['close'] - df['open']
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']

        # Add 'candle_range' only if legacy_compatibility is False
        if not legacy_compatibility:
            df['candle_range'] = df['high'] - df['low']

        
    def calculate_sma(self, df):
        """
        Calculate Simple Moving Averages (SMA).
        """
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
    def calculate_atr(self, df):
        """
        Calculate Average True Range (ATR).
        """
        high_low = df['high'] - df['low']
        high_prev_close = (df['high'] - df['close'].shift()).abs()
        low_prev_close = (df['low'] - df['close'].shift()).abs()
        true_range = high_low.to_frame('hl').join(high_prev_close.to_frame('hpc')).join(low_prev_close.to_frame('lpc')).max(axis=1)
        df['atr_14'] = true_range.ewm(alpha=1/14, adjust=False).mean()



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
        df['macd_lag_1'] = df['macd'].shift(1)



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

    def calculate_regime(self, df, slope_threshold=0.02):
        """
        Classifies market regime using:
        - 'bull' → SMA 10 crossing above SMA 50 & slope confirmation
        - 'bear' → SMA 10 crossing below SMA 50 & slope confirmation
        - 'high_vol' → High ATR (Volatile Market)
        - 'low_liquidity' → Low Trading Volume
        - 'neutral' → No strong trend

        Args:
            df (pd.DataFrame): Data containing SMA and ROC (Rate of Change).
            slope_threshold (float): Minimum slope change to confirm trend.

        Returns:
            pd.DataFrame: Updated DataFrame with 'regime' column.
        """

        # Compute SMA crossovers (previous value vs. current value)
        df["sma_crossover_up"] = (df["sma_10"].shift(1) < df["sma_50"].shift(1)) & (df["sma_10"] > df["sma_50"])
        df["sma_crossover_down"] = (df["sma_10"].shift(1) > df["sma_50"].shift(1)) & (df["sma_10"] < df["sma_50"])

        # Compute SMA slopes (Rate of Change over 10 periods)
        df["sma_10_slope"] = df["sma_10"].pct_change(periods=10)
        df["sma_50_slope"] = df["sma_50"].pct_change(periods=10)

        # Define market regimes
        df["regime"] = np.select(
            [
                df["atr_14"] > df["atr_14"].quantile(0.8),  # High volatility
                df["sma_crossover_up"] & (df["sma_10_slope"] > slope_threshold),  # Bullish crossover & slope
                df["sma_crossover_down"] & (df["sma_10_slope"] < -slope_threshold),  # Bearish crossover & slope
                df["volume_ma_20"] < df["volume_ma_20"].quantile(0.3),  # Low liquidity
            ],
            ["high_vol", "bull", "bear", "low_liquidity"],
            default="neutral"
        )

        # Drop intermediate columns (optional)
        # df.drop(columns=["sma_crossover_up", "sma_crossover_down", "sma_10_slope", "sma_50_slope"], inplace=True)

        return df

