# Existing BTC model: offline audit

## Identification

The intended legacy 1-minute production artifact is
`best_model_10candles_reduced_1min.joblib`, because `bot-1m.py` loads it
directly and fixes the signal thresholds at +0.09 (buy) and -0.14 (sell).
This selection is based on deployment wiring, not 2025 performance.

It is a 50-tree `RandomForestRegressor` (`max_depth=10`,
`max_features=15`, `bootstrap=False`) with no scaler. Its embedded feature
order is the 23 columns enforced by `offline_btc_evaluation.py`. The formulas
are copied from the explicit feature cell in `bot.ipynb`; the current parent
`BatchFeatures.py` is not used because it changes names and several formulas.

The training notebook `linear.ipynb` labels its target as:

`(close[t-10] - close[t]) / close[t] * 100`

That is a backward-looking 10-bar return, not a future return. `roc_10` is an
algebraic transform of the same two prices and accounts for 74.89% of the
model's feature importance. The high notebook fit is therefore not evidence
of forecast skill. The artifact filename says 10 candles and 1 minute; it has
no sequence input, so its feature lookback is 50 bars (from SMA-50), not an
LSTM lookback window.

Exact training dates cannot be recovered: they are absent from the artifact,
and the notebook's source `BTCUSDT_1min` database is not present. The original
test period was the final 20% of that missing dataset. Its thresholds were
then optimized on that same test period, so the reported test trading result
overlaps model selection. The notebook additionally traded `Predicted Price`
rather than an observed/fillable price and applied no fees or slippage.

## Other plausible artifacts

| Artifact/family | Preprocessing and order | Horizon / sequence | Status |
|---|---|---|---|
| `best_model_12candles.joblib` | No scaler; same embedded 23-feature order | filename/live comments say 12 × 5m | Loaded by `bot.py`; training dates absent |
| `grid_search_results_1min_reduced.pkl` | Pipeline grid object; embedded 23-feature order | notebook target is backward 10 bars | Supplies the selected RF hyperparameters |
| `grid_search_results_1min_nn10.pkl` and negative variant | Pipeline scaler plus RF; embedded 34-feature order | 10 × 1m per names/notebook | Candidate search objects, not deployed final model |
| `grid_rf_search_results_1min_ema5_nn10.pkl` | Pipeline; embedded 51-feature order | future EMA-5, 10 × 1m | Search object; not deployed |
| `grid_xgb_search_results_1min_ema5_nn10.pkl` | Pipeline; same 51 features | future EMA-5, 10 × 1m | Best score is NaN; reject as selected model |
| `lstm_10candles_1min.joblib` + `lstm_scaler.pkl` | 61-column scaler, but model accepts 52 | 60-bar sequence; nominal 10-bar target | Incompatible pair; do not substitute columns |
| `lstm_n10candles_1min_ema5.joblib` + `lstm_scaler_em5_n10.pkl` | exact 61-column match including target lags 1..9 | 60-bar sequence; negative future EMA-5 return over 10 bars | Plausible and wired in `bot-1m-rnn-ema-5.py`; target-lag availability makes live inference suspect |
| `lstm_ema_10candles_1min.joblib` + `lstm_ema_scaler.pkl` | exact 54-column match | 60-bar sequence; notebook EMA/raw blended target | Plausible and wired in `bot-1m-rnn-ema.py`; live target code has a sign error (`nn=-10` then `shift(-nn)`) |
| `lstm_10candles_1sec_ema5.joblib` + `lstm_scaler_ema5_1sec.pkl` | model accepts 56, scaler emits 71 | 60-second sequence; nominal 5-bar target in notebook despite filename | Incompatible pair; notebook training later reaches NaNs |
| remaining LSTMs/scalers | Several 51/52-feature models and 17/30/54-feature scalers | notebooks vary between 1, 5, and 10 bars | No exact, unambiguous deployed pairing |

No LSTM artifact stores its training dates. Scaler means are not dates and
were not used to guess them.

## Environment and offline execution

Use conda environment `ml-2`: Python 3.11.11, joblib 1.4.2, NumPy 2.0.2,
pandas 2.2.3, scikit-learn 1.6.0, TensorFlow 2.18.0, XGBoost 2.1.3. This aligns
with `requirements-py3-11.txt` and the `ml-2` notebook metadata. No packages
were installed or upgraded. The active `binbot` environment is Python 3.10
with scikit-learn 1.7.2 and lacks TensorFlow/XGBoost, so it is not the intended
reproduction environment.

Example (offline, read-only database, isolated new output directory):

```bash
conda run -n ml-2 python offline_btc_evaluation.py \
  --db ../db/BTCUSDT_1m_52weeks.db \
  --output-dir offline_evaluation/rf_1m_2025_2026
```

The evaluator forms a signal only after bar *t* closes and fills at bar
*t+1* open. It reports gross and net performance on exactly the same round
trips. Fees are in basis points of notional **per side** (default 10 bps =
0.10% added to entry notional and deducted from exit notional); default
slippage is 0 bps per side.

The available `binance_data.db` covers 2024-01-01 through 2024-12-01 but has
5-minute density, so it is not silently used as the missing 1-minute training
database. The available 1-minute annual database covers 2025-10-09 through
2026-10-08 and is strictly a later diagnostic period, not a basis for model
selection.

## Later-period diagnostic result

On the available 2025-10-09 16:57 UTC through 2026-10-08 16:56 UTC 1-minute
data, the fixed deployed thresholds produced 9,893 round trips (including one
forced final exit). Average gross return was 0.0028837% per round trip, or
0.2884 bps. Gross compounded return was +25.4787%. With 10 bps per side and
zero modeled slippage, net compounded return on those exact same trades was
-99.9999997%. The enormous turnover, rather than a change in selected trades,
causes the gross/net divergence.

The model achieves R-squared 0.99745 against its documented backward-looking
target on this later data. That is a leakage diagnostic, not forecast
performance. These dates cannot overlap the unknown training dates based on
available evidence alone; however, they do not overlap the 2024 data file and
they were not used here for model or threshold selection.
