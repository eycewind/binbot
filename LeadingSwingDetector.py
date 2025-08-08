import numpy as np
import pandas as pd

class LeadingSwingDetector:
    def __init__(self, left_bars=3):
        self.left_bars = left_bars
        # Buffer to store last left_bars highs and lows for lookback continuity
        self.high_buffer = []
        self.low_buffer = []
        # Store the last processed DataFrame index so that we know where to continue
        self.last_processed_index = None
        # Store last processed swing_high and swing_low arrays for continuity
        self.swing_high_list = []
        self.swing_low_list = []

    def detect_leading_swings_incremental(self, df_new):
        """
        Incrementally detect swing highs and lows using only left_bars lookback on new incoming data.
        Maintains state internally so this can be called repeatedly on new data batches.

        Args:
            df_new (pd.DataFrame): New incoming data with 'high' and 'low' columns, datetime indexed.

        Returns:
            pd.DataFrame: Input df_new with 'swing_high' and 'swing_low' columns updated for new bars.
        """
        highs_new = df_new['high'].values
        lows_new = df_new['low'].values

        total_len = len(highs_new)

        # Initialize swing arrays for the new data (default False)
        swing_highs_new = np.full(total_len, False, dtype=bool)
        swing_lows_new = np.full(total_len, False, dtype=bool)

        # Combine buffer and new data highs/lows for lookback check
        extended_highs = np.array(self.high_buffer + highs_new.tolist())
        extended_lows = np.array(self.low_buffer + lows_new.tolist())

        # Start checking swing from position = len(buffer) to end (relative to extended array)
        start_pos = len(self.high_buffer)

        for i in range(start_pos, len(extended_highs)):
            # Need at least left_bars behind
            if i - self.left_bars < 0:
                continue
            # Check swing high condition: current high is greater than all left_bars previous highs
            if all(extended_highs[i] > extended_highs[i - j - 1] for j in range(self.left_bars)):
                swing_highs_new[i - start_pos] = True
            # Check swing low condition similarly
            if all(extended_lows[i] < extended_lows[i - j - 1] for j in range(self.left_bars)):
                swing_lows_new[i - start_pos] = True

        # Update buffer for next incremental call with last left_bars highs/lows from extended arrays
        if len(extended_highs) >= self.left_bars:
            self.high_buffer = extended_highs[-self.left_bars:].tolist()
            self.low_buffer = extended_lows[-self.left_bars:].tolist()
        else:
            # In rare cases if data less than left_bars
            self.high_buffer = extended_highs.tolist()
            self.low_buffer = extended_lows.tolist()

        # Update last processed index - assume index is datetime and append continuously
        if df_new.index.size > 0:
            self.last_processed_index = df_new.index[-1]

        # Assign swing arrays to df_new new columns
        df_new = df_new.copy()  # Avoid modifying original outside
        df_new['swing_high'] = swing_highs_new
        df_new['swing_low'] = swing_lows_new

        return df_new
