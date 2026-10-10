import unittest

import pandas as pd

from run_baseline_rebuild import FEE_RATE, MODEL_BREAK_EVEN, accepted_trades


class RunnerContractTests(unittest.TestCase):
    def test_break_even_is_exact_for_two_sided_fee_accounting(self):
        net_multiplier = (1 + MODEL_BREAK_EVEN) * (1 - FEE_RATE) / (1 + FEE_RATE)
        self.assertAlmostEqual(net_multiplier, 1.0, places=15)

    def test_exit_precedes_reentry_at_same_open(self):
        index = pd.date_range("2026-01-01", periods=22, freq="min", tz="UTC")
        frame = pd.DataFrame(index=index[:11])
        frame["entry_time"] = index[1:12]
        frame["exit_time"] = index[11:]
        frame["entry_open"] = 100.0
        frame["exit_open"] = 101.0
        signal = pd.Series(False, index=frame.index)
        signal.iloc[0] = True
        signal.iloc[10] = True
        trades = accepted_trades(frame, signal, "test")
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades.iloc[0].exit_time, trades.iloc[1].entry_time)


if __name__ == "__main__":
    unittest.main()
