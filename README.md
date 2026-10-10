# binbot

Research and archived trading code for cryptocurrency market data. The active
BTC forecasting baseline is price-only and offline; sentiment and historical
live-bot code are separate and must not be imported into its evaluation path.

## Active BTC baseline

- [Frozen rebuild contract](old-work/BASELINE_REBUILD_CONTRACT.md)
- [Data inventory and chronological partitions](old-work/BTC_DATA_INVENTORY.md)
- [Baseline evaluation results](old-work/BASELINE_REBUILD_RESULTS.md)
- [Legacy BTC model audit](old-work/BTC_MODEL_AUDIT.md)
- [Repository map](REPO_MAP.md)
- [Proposed staged reorganization](RESTRUCTURE_PLAN.md)

The corrected baseline uses conda environment `ml-2` (verified with
scikit-learn 1.6.0), reads local one-minute BTC data, and has no exchange or
order-submission dependency.

From the repository root:

```bash
conda run -n ml-2 python -m pytest -q old-work/test_baseline_rebuild.py
cd old-work
conda run -n ml-2 python run_baseline_rebuild.py
```

The second command regenerates evaluation outputs. Preserve the committed
`old-work/baseline_rebuild_results/` run before doing so, or direct experimental
work to a separately named output directory after adding an explicit CLI option.
The test partition is reused and must not be described as untouched.

## Repository safety

- Historical `old-work/bot*.py` files may connect to an exchange or submit
  orders. They are not offline evaluation entry points.
- Root research notebooks may contain fallback data-download behavior.
- Do not substitute model/scaler artifacts based on similar filenames; consult
  the legacy audit and recorded hashes.
- The 2025 sentiment dataset is not part of the price-only baseline and does not
  overlap its validation/test periods.
- Large databases, environments, artifacts, and generated results do not yet
  have a unified storage policy. The proposed policy and migration gates are in
  `RESTRUCTURE_PLAN.md`.
