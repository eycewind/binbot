# BTC market-data inventory for the corrected baseline

Inventory performed read-only in conda environment `ml-2`. A gap is a
timestamp difference greater than the dataset's median cadence.

| File | Coverage UTC | Rows | Cadence | Gaps / duplicates | Baseline use |
|---|---|---:|---:|---:|---|
| `../db/BTCUSDT_1m_52weeks.db` | 2025-10-09 16:57 to 2026-10-08 16:56 | 524,160 | 60 s | 0 / 0 | Primary |
| `BTCUSDT_1m_dry_run.db` | 2026-01-19 15:02 to 2026-01-24 15:01 | 7,200 | 60 s | 0 / 0 | Exact subset/overlap; exclude |
| `binance_data.db` | 2024-01-01 00:00 to 2024-12-01 00:00 | 96,481 | 300 s | 0 / 0 | Wrong cadence; exclude |
| `../db/BTCUSDT_5m_52weeks.db` | 2025-01-25 15:20 to 2026-01-24 15:15 | 104,832 | 300 s | 0 / 0 | Wrong cadence; exclude |
| `../BTCUSDT_5m_52weeks.db` | 2025-10-11 10:25 to 2026-10-10 10:20 | 104,832 | 300 s | 0 / 0 | Wrong cadence; exclude |
| `BTCUSDT_1sec_dry_run.db` | 2025-01-09 17:08:59 to 2026-01-24 14:52 | 247,680 | nominal 1 s | 74,880 gaps; maximum 28,158,242 s; 0 duplicates | Discontinuous/wrong cadence; exclude |

The primary file has exactly 364 continuous days and no missing expected
minutes. Its first and last UTC days are partial, so the proposed partitions
in `BASELINE_REBUILD_CONTRACT.md` use 363 complete days. The dry-run minute
file is not extra evidence because it lies wholly inside the primary file's
coverage. The two annual five-minute files are separate snapshots and are not
upsampled or mixed into the one-minute baseline.

The source database named `BTCUSDT_1min` by the legacy training notebook is
absent. Consequently, legacy training dates remain unknown. The full primary
one-minute period has already been inspected by `offline_btc_evaluation.py`;
none of its proposed partitions can honestly be called untouched.
