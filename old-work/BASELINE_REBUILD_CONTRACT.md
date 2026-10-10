# Corrected BTC forecasting baseline contract

Status: specified and alignment-tested; no model has been trained under this
contract. The legacy artifacts and `offline_evaluation/` outputs are inputs to
the audit only and remain unchanged.

## 1. Leakage finding and corrected forecasting question

`linear.ipynb` lines 148-149 (cell 7) and 1899-1900 (cell 54) set `nn = 10`
and define:

```python
legacy_target_t = (close[t-10] - close[t]) / close[t] * 100
```

`bot.ipynb` line 250 (cell 11) defines:

```python
roc_10_t = (close[t] - close[t-10]) / close[t-10] * 100
```

Let `r = roc_10_t / 100`, so `close[t] = close[t-10] * (1+r)`. Then:

```text
legacy_target_t = -roc_10_t / (1 + roc_10_t / 100)
```

Thus the legacy model reconstructs a past return already present in its input;
it does not predict a future return. This identity is tested exactly in
`test_baseline_rebuild.py`.

The corrected question is fixed before training: after the one-minute bar at
decision timestamp *t* has closed, forecast the executable BTC return from
the next bar's open to the open exactly ten minutes later.

The model target is a **decimal return**, not percent or basis points:
`future_return_10m = open[t+11] / open[t+1] - 1`. For example, `0.002` means
0.20% or 20 basis points.

The feature order is frozen exactly as follows:

1. `return_1m = close[t] / close[t-1] - 1`
2. `return_5m = close[t] / close[t-5] - 1`
3. `return_10m = close[t] / close[t-10] - 1`
4. `range_fraction = (high[t] - low[t]) / close[t]`
5. `body_fraction = (close[t] - open[t]) / open[t]`
6. `volume_log1p = log(1 + volume[t])`
7. `volume_ratio_20 = volume[t] / mean(volume[t-19:t])`
8. `volatility_20 = sample_std(return_1m[t-19:t])`

No feature may be added, removed, reordered, or transformed in this run.

## 2. Clock, target, and execution

- Bars are one-minute UTC bars whose index is the bar-open timestamp.
- The prediction is made only after bar *t* closes. Features may use OHLCV at
  *t* and earlier, never any value from *t+1* onward.
- Earliest entry is `open[t+1]`.
- The fixed horizon is ten minutes. There is no horizon search.
- Exit is `open[t+11]`, exactly ten minutes after entry.
- Decimal gross label and trade return are
  `R_gross[t] = open[t+11] / open[t+1] - 1`.
- Both timestamps must exist exactly one and eleven minutes after *t*;
  otherwise the sample is invalid. Positional adjacency alone is insufficient.
- Targets are generated independently and are never inputs to feature
  generation or preprocessing.

Each evaluation partition starts flat with 10,000 USDT. Long-only position
sizing is fixed at 100% of available strategy cash per accepted signal,
including entry costs: quantity is `cash / (entry_execution_price * (1+f))`.
At most one position may exist. A signal is accepted only when flat; all
signals while holding are ignored. Every accepted position is exited at its
scheduled `open[t+11]`; predictions do not cause early exits. If an exit and a
new signal fill share a timestamp, process the scheduled exit first, then
allow the new entry at that same open. Both round trips pay their applicable
side fees, including an exit fee followed by a new entry fee at that timestamp.
This makes overlapping-signal handling deterministic.

Entry slippage is added to the next open and exit slippage is subtracted from
the exit open. Initial reporting uses **zero slippage per side**, explicitly
because no trustworthy historical spread/impact series is present. Fees use
10 basis points (0.10%) of notional per side unless a documented venue tier is
supplied before training. For per-side fee `f` and slippage `s`:

```text
P_entry = open[t+1]  * (1+s)
P_exit  = open[t+11] * (1-s)
R_net   = (P_exit / P_entry) * ((1-f)/(1+f)) - 1
```

Gross and net statistics must use the identical accepted trade list.

## 3. Data partitions and preprocessing

The proposed boundaries use complete UTC days from the clean annual 1-minute
database:

| Partition | Decision timestamps | Raw bars | Label-valid before feature warm-up |
|---|---|---:|---:|
| Train | 2025-10-10 00:00 to 2026-05-15 00:00 exclusive | 312,480 | 312,469 |
| Validation | 2026-05-15 00:00 to 2026-07-27 00:00 exclusive | 105,120 | 105,109 |
| Test | 2026-07-27 00:00 to 2026-10-08 00:00 exclusive | 105,120 | 105,109 |

The final eleven decision rows in each partition are purged because their
entry/exit path crosses the exclusive boundary. Feature warm-up NaNs are
dropped only after partition membership and label validity are established.
Features may carry historical context across a boundary because that context
was already observable, but labels may never cross one.

Any learned imputation, centering, or scaling is fit on training rows only.
It is then frozen for validation and test. Any trading threshold selection is
performed on validation only and frozen exactly once before test. The legacy
offline evaluation already inspected the entire 2025-10-09 through 2026-10-08
period, including the proposed test dates. Therefore this is a **reused
evaluation holdout**, not an untouched test set; no untouched 1-minute period
is currently available.

## 4. Fixed initial comparisons

No complexity or parameter search is authorized in the initial run:

1. **No trade:** equity remains constant; zero trades and zero turnover.
2. **Momentum:** enter when the causal 10-minute close-to-close return is
   strictly greater than +0.10%; otherwise stay flat. Hold accepted entries
   for the same fixed ten-minute execution horizon. The +0.10% rule is fixed
   now and is not selected from validation results.
3. **Modest model:** scikit-learn `StandardScaler` followed by `Ridge(alpha=1.0,
   fit_intercept=True)`, using the fixed ordered causal features above. The
   scaler is fit on training rows only and frozen with the fitted Ridge model.
   With `f=0.001` per side and zero slippage, exact break-even gross return is
   `(1+f)/(1-f)-1 = 0.002002002002...` (0.2002002%, 20.02002 bps). Enter only
   when the predicted executable gross return is strictly greater than that
   fixed value. It is derived from costs, not selected on validation or test.
   These are initial settings, not tuned settings.

For each comparison report gross and net compounded return, annualized
turnover (`sum(entry_notional + exit_notional) / average_equity`), accepted
round-trip count, maximum drawdown from a minute-indexed marked-to-market
equity curve, arithmetic average gross and net return per accepted trade, and
exposure fraction. Also report predictive MAE and RMSE for the model, but do
not use test metrics to change the model or rule.

Zero slippage is an explicitly optimistic baseline assumption. Sensitivities
at fixed 1 and 2 bps per side must reprice the exact same accepted trades;
they must not alter predictions, signals, entries, or the preferred scenario.
Total fees are the sum of quote-currency entry and exit fees along the
compounded 10,000-USDT equity path. Turnover uses executed entry plus exit
notional divided by average minute-marked equity, annualized by the exact
evaluation duration. Maximum drawdown uses that same marked-to-market equity
curve; open BTC is marked at each bar close and net liquidation value includes
the exit fee and selected slippage.

## 5. Gate before training

Training may begin only after this contract and the inventory are accepted.
The first run must use these exact partitions, feature definitions, target,
comparisons, costs, and settings. Changes require a new contract version and
must not be justified using the reused test segment.

## 6. Pre-training verification result

Run in verified `ml-2` with:

```bash
conda run -n ml-2 python -m unittest -v test_baseline_rebuild.py
```

All five tests pass: exact legacy algebra, executable target indexing,
future-mutation feature invariance, gap rejection, and boundary-label purge.
On the real primary database the algebraic identity's maximum absolute
floating-point error is `1.1100495522775589e-14`. Usable counts after target
validation and feature availability are 312,469 train, 105,109 validation,
and 105,109 test. No estimator was fit to obtain these results.
