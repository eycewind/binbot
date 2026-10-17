# E0.3a: non-invasive walk-state audit

Purpose: instrument the frozen E0.2 `detect_walk_states` without changing its states.

`audit_walk_states` first calls the baseline implementation, then independently replays it and asserts exact parity of the state timeline and transition events.

**Important limitations:** this is implementation-equivalence validation, not empirical predictive accuracy. Confirmation is only known on its confirmation candle; `candidate_since` is descriptive historical metadata and must not be used to backdate a confirmed trading signal. The existing E0.2 definitions (3 consecutive near/slope qualifications, 2 distance exits) are retained, including their limitations. The historical market labels and E04 false-alarm measures remain future tasks.

The analysis notebook requires the user to connect their verified 5-minute `bars` loader. Use the full input history including warmup, not a November-only cropped dataset. The notebook audits the full year and uses November only for optional inspection.


E0.3b walk outcome study
========================
1. Place alongside the new notebook in your research/sentiment_price directory.
2. Reuse your verified E03 loader for the 5-minute `bars` dataframe.
3. Run `python -m pytest -q test_e03b.py` then execute the notebook.
4. There is no detector tuning, trading backtest, or actual BTC outcomes in this package.
5. Snapshot dependence and censored observations require later uncertainty analysis.

# E0.3c — Detector validation, preliminary

Install numpy, pandas, matplotlib, jupyter, pytest. Unzip into research/sentiment_price.
Run `python -m pytest -q test_e03c.py`.
Launch `E03c_Detector_Validation.ipynb`; paste your **verified 5m bars loader** into first setup cell.

Uses unchanged E0.2 walk detector as V0. All other variants generate causal alarms.
Reference price impulses use ATR, not Bollinger bands, but their onset is retrospective.
The output is exploratory detector-vs-proxy comparison, not objective ground-truth accuracy,
CFAR-calibrated Pfa, or proof of profitable trading. Freeze variants before holdout testing.
