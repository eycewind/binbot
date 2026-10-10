# Frozen price-only baseline results

This is the single run authorized by `BASELINE_REBUILD_CONTRACT.md`. No
feature, horizon, model, threshold, or parameter search was performed. The
Ridge pipeline was fit once on 312,469 training samples. Validation and the
**reused** test partition were evaluated independently, each starting with
10,000 USDT and no position.

The model signal hurdle was fixed before results at the exact zero-slippage
fee break-even return, `0.002002002002001957` decimal (20.02002 bps). The
momentum comparator was fixed at `return_10m > 0.001`. Fees are 10 bps per
side. Zero slippage is optimistic; 1 and 2 bps-per-side results are mechanical
sensitivities on identical trades and signals.

## Predictive results

Metrics are decimal-return units.

| Partition | Predictor | MAE | RMSE |
|---|---|---:|---:|
| Validation | Ridge | 0.0011731644 | 0.0017944022 |
| Validation | Training-target mean | 0.0011724562 | 0.0017940727 |
| Reused test | Ridge | 0.0009764893 | 0.0015367103 |
| Reused test | Training-target mean | 0.0009752507 | 0.0015356834 |

Ridge is slightly worse than the training-mean predictor on both partitions.
Predictive error is reported separately and does not replace trading results.

## Validation trading results

| Strategy | Trades | Exposure | Gross return | Net return, 0 bps slip | Gross / net max DD | Avg gross / net trade | Fees, USDT | Annualized turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No trade | 0 | 0% | 0% | 0% | 0% / 0% | n/a | 0.00 | 0.00× |
| Momentum | 3,832 | 36.4536% | -19.4996% | -99.9622% | -22.3960% / -99.9623% | -0.005445% / -0.205234% | 9,335.83 | 34,138.57× |
| Ridge | 2 | 0.0190% | +0.8268% | +0.4243% | -0.3458% / -0.3458% | +0.413290% / +0.212664% | 40.16 | 20.03× |

Net return sensitivity on the same trades:

| Strategy | 0 bps/side | 1 bp/side | 2 bps/side |
|---|---:|---:|---:|
| No trade | 0% | 0% | 0% |
| Momentum | -99.9622% | -99.9824% | -99.9918% |
| Ridge | +0.4243% | +0.3841% | +0.3439% |

## Reused-test trading results

| Strategy | Trades | Exposure | Gross return | Net return, 0 bps slip | Gross / net max DD | Avg gross / net trade | Fees, USDT | Annualized turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No trade | 0 | 0% | 0% | 0% | 0% / 0% | n/a | 0.00 | 0.00× |
| Momentum | 3,315 | 31.5354% | +3.0847% | -99.8639% | -12.5453% / -99.8639% | +0.001097% / -0.198705% | 10,138.58 | 29,959.83× |
| Ridge | 0 | 0% | 0% | 0% | 0% / 0% | n/a | 0.00 | 0.00× |

Net return sensitivity on the same trades:

| Strategy | 0 bps/side | 1 bp/side | 2 bps/side |
|---|---:|---:|---:|
| No trade | 0% | 0% | 0% |
| Momentum | -99.8639% | -99.9299% | -99.9639% |
| Ridge | 0% | 0% | 0% |

The two validation Ridge trades are too few to establish forecasting value,
and the fixed Ridge rule generated no reused-test trades. The fixed momentum
rule's high turnover makes it economically unusable at the contracted fee.
These observations do not authorize tuning.

## Reproducibility and scope

`baseline_rebuild_results/frozen_v1/config.json` records the ordered features,
target, rules, environment, source hashes, input database hash, and fitted
pipeline hash. `ridge_pipeline.joblib` contains both the training-fitted
`StandardScaler` and `Ridge(alpha=1)` model. Full predictions and one ledger
per strategy/partition are preserved in that output directory. Gross and net
statistics use identical ledgers; only fee/slippage repricing changes.

The current sentiment data is a 2025 dataset. It does not overlap validation
(2026-05-15 through 2026-07-26) or reused test (2026-07-27 through 2026-10-07).
No sentiment file was accessed or integrated for this run.
