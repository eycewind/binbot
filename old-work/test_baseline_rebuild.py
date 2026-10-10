import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from baseline_rebuild import (
    Partition,
    legacy_target_from_roc,
    make_causal_features,
    make_forward_target,
    partition_mask,
)


def sample_bars(rows=120):
    index = pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC")
    close = pd.Series(100 + np.arange(rows) * 0.1, index=index)
    return pd.DataFrame(
        {"open": close - 0.02, "high": close + 0.05, "low": close - 0.05,
         "close": close, "volume": 10 + np.arange(rows)}, index=index
    )


class BaselineRebuildTests(unittest.TestCase):
    def test_legacy_target_is_exact_transform_of_roc_10(self):
        bars = sample_bars()
        roc = bars.close.pct_change(10) * 100
        notebook_target = (bars.close.shift(10) - bars.close) / bars.close * 100
        assert_series_equal(legacy_target_from_roc(roc), notebook_target)

    def test_target_uses_next_open_and_open_t_plus_11(self):
        bars = sample_bars()
        target = make_forward_target(bars)
        t = bars.index[30]
        self.assertEqual(target.loc[t, "entry_time"], bars.index[31])
        self.assertEqual(target.loc[t, "exit_time"], bars.index[41])
        expected = bars.open.iloc[41] / bars.open.iloc[31] - 1
        self.assertAlmostEqual(target.loc[t, "future_return_10m"], expected)

    def test_changing_future_bars_cannot_change_earlier_features(self):
        bars = sample_bars()
        cutoff = bars.index[70]
        before = make_causal_features(bars).loc[:cutoff]
        changed = bars.copy()
        changed.loc[changed.index > cutoff, ["open", "high", "low", "close", "volume"]] *= 100
        after = make_causal_features(changed).loc[:cutoff]
        assert_frame_equal(before, after)

    def test_gap_invalidates_cross_gap_label(self):
        bars = sample_bars().drop(sample_bars().index[35])
        targets = make_forward_target(bars)
        self.assertTrue(pd.isna(targets.loc[bars.index[30], "future_return_10m"]))

    def test_partition_purges_forward_labels_crossing_boundary(self):
        bars = sample_bars()
        targets = make_forward_target(bars)
        boundary = bars.index[80]
        part = Partition("train", bars.index[0], boundary)
        mask = partition_mask(targets, part)
        self.assertTrue(mask.loc[bars.index[68]])   # exit at position 79
        self.assertFalse(mask.loc[bars.index[69]])  # exit equals exclusive boundary


if __name__ == "__main__":
    unittest.main()
