import unittest
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from e01_visualize import episode_diagnostics, plot_episode_boundaries

class E01Tests(unittest.TestCase):
    def setUp(self):
        self.episodes = pd.DataFrame([dict(side='Support', episode_id='S0001',
            start=pd.Timestamp('2025-11-01T01:00Z'),
            observation_end=pd.Timestamp('2025-11-01T02:00Z'),
            status='REARMED', n_bars=2, n_contacts=2)])

    def test_durations(self):
        e, s = episode_diagnostics(self.episodes, '15min')
        self.assertEqual(e.duration_hours.iloc[0], .5)
        self.assertEqual(s.loc['Support', 'median_duration_hours'], .5)
        self.assertEqual(e.elapsed_close_to_close_hours.iloc[0], 1)

    def test_plot(self):
        idx = pd.date_range('2025-11-01T00:00Z', periods=6, freq='1h')
        bb = pd.DataFrame(dict(close=[100]*6, bb_middle=[100]*6,
            bb_lower=[95]*6, bb_upper=[105]*6), index=idx)
        bb.attrs['resolution'] = '1h'
        fig, axes, selected = plot_episode_boundaries(bb, self.episodes,
            '2025-11-01', '2025-11-02')
        self.assertEqual(len(selected), 1)
        fig.clf()

if __name__ == '__main__':
    unittest.main()
