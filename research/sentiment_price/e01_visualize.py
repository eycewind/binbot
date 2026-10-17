"""E0.1 diagnostics for Bollinger episode detector.

Does not alter event detection or assign breakout/rejection labels.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def episode_diagnostics(episodes, resolution):
    """Return episode summary with comparable elapsed-time measures.

    n_bars includes the starting candle, so elapsed observation from candle
    OPEN to the final observed candle CLOSE is n_bars * bar duration.
    """
    step = pd.Timedelta(resolution)
    if step <= pd.Timedelta(0):
        raise ValueError('resolution must be positive')
    e = episodes.copy()
    e['duration_hours'] = e['n_bars'] * step / pd.Timedelta('1h')
    e['elapsed_close_to_close_hours'] = (
        pd.to_datetime(e['observation_end'], utc=True) - pd.to_datetime(e['start'], utc=True)
    ) / pd.Timedelta('1h')
    summary = e.groupby('side', sort=True).agg(
        episodes=('episode_id', 'size'),
        median_bars=('n_bars', 'median'),
        median_duration_hours=('duration_hours', 'median'),
        max_duration_hours=('duration_hours', 'max'),
        median_contacts=('n_contacts', 'median'),
        timeouts=('status', lambda s: int(s.eq('TIMEOUT').sum())),
        right_censored=('status', lambda s: int(s.eq('RIGHT_CENSORED').sum())),
    )
    return e, summary


def plot_episode_boundaries(bb, episodes, start, end, side='Support',
                            figsize=(15, 8), max_labels=18):
    """Plot price/bands and episode intervals, restricted to a UTC time range.

    Interval shading uses recorded first/last candle close timestamps. The
    lower strip distinguishes multiple adjacent episodes that overlap visually
    in a downtrend. Only previously computed episodes are plotted.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize('UTC') if start.tzinfo is None else start.tz_convert('UTC')
    end = end.tz_localize('UTC') if end.tzinfo is None else end.tz_convert('UTC')
    if start >= end:
        raise ValueError('start must precede end')
    if side not in ('Support', 'Resistance'):
        raise ValueError('side must be Support or Resistance')
    step = pd.Timedelta(bb.attrs['resolution'])
    df = bb.sort_index().loc[(bb.index >= start) & (bb.index < end)].copy()
    if df.empty:
        raise ValueError('No candles in selected window')
    e = episodes.loc[
        episodes['side'].eq(side)
        & (pd.to_datetime(episodes['start'], utc=True) < end)
        & (pd.to_datetime(episodes['observation_end'], utc=True) >= start)
    ].sort_values('start').copy()

    fig, (ax, timeline) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [4, 1]},
        constrained_layout=True
    )
    ax.plot(df.index, df['close'], color='black', linewidth=1.1, label='Close')
    ax.plot(df.index, df['bb_middle'], color='tab:orange', linewidth=1, label='BB middle')
    ax.plot(df.index, df['bb_upper'], color='tab:blue', linewidth=0.85, label='BB upper')
    ax.plot(df.index, df['bb_lower'], color='tab:blue', linewidth=0.85, label='BB lower')
    ax.fill_between(df.index, df['bb_lower'].to_numpy(), df['bb_upper'].to_numpy(),
                    alpha=0.06, color='tab:blue')

    status_colors = {'REARMED': 'tab:green', 'TIMEOUT': 'tab:red',
                     'RIGHT_CENSORED': 'tab:purple', 'DATA_GAP_CENSORED': 'tab:gray'}
    for j, (_, ev) in enumerate(e.iterrows()):
        s = max(pd.Timestamp(ev['start']) - step, start)
        z = min(pd.Timestamp(ev['observation_end']), end)
        color = status_colors.get(ev['status'], 'tab:gray')
        ax.axvspan(s, z, facecolor=color, alpha=0.10)
        timeline.axvspan(s, z, facecolor=color, alpha=0.46, edgecolor=color, lw=0.4)
        ts = pd.Timestamp(ev['start'])
        if start <= ts < end:
            timeline.axvline(ts, color=color, linewidth=0.8)
            if j < max_labels:
                timeline.text(ts, 0.5, str(ev['episode_id']), rotation=90,
                              fontsize=7, ha='right', va='center')
    timeline.set_ylim(0, 1)
    timeline.set_yticks([])
    timeline.set_ylabel('Episodes')
    timeline.grid(axis='x', alpha=0.15)
    ax.grid(alpha=0.16)
    ax.set_ylabel('USDT')
    ax.set_title(f'BTCUSDT Bollinger bands + {side.lower()} episode boundaries (E0.1)')
    ax.legend(loc='upper right')
    timeline.set_xlabel('Candle close time (UTC)')
    timeline.legend(handles=[Patch(facecolor=c, alpha=0.6, label=s)
                             for s, c in status_colors.items()],
                    loc='upper left', ncol=4, fontsize=8)
    ax.set_xlim(start, end)
    return fig, (ax, timeline), e
