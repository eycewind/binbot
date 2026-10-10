# Repository Map

This document is a read-only structural audit of the repository as inspected on
2026-10-10. It describes observed relationships; it does not assign chronology
from filenames or filesystem timestamps.

## Repository boundary and status

- Git root: `/media/ata/Data/personal/binbot`
- Branch at inspection: `btc-sentiment-classifier-v3`
- The requested working directory, `old-work/`, is a directory in that Git
  repository. Its empty `.git/` directory is not a repository: Git resolves
  `old-work/` to the parent root.
- No `AGENTS.md` or other repository-specific agent instruction file was found.
- No nested Git worktrees were found beneath the root.
- The only observed symlinks are Python interpreter/library links under the
  checked-in `.venv/`; no market-data or project-directory symlinks were found.
- Directories adjacent to `binbot/` under `/media/ata/Data/personal/` are
  separate projects and are out of scope. Examples include `binance-v2/`,
  `psx/`, `us-stocks/`, `agri-ai/`, and several backup/personal directories.
- The worktree contained pre-existing modified and untracked files. In
  particular, `.vscode/settings.json`, `nushur.ipynb`, `.envrc`, several market
  databases, sentiment directories/archives, and `old-work/offline_evaluation/`
  were not created or altered by this documentation audit.

The root `.gitignore` is minimal. It excludes `.env-secret`, `__pycache__`,
`.vscode`, and `:Zone.Identifier`, but does not yet establish policies for
virtual environments, databases, model artifacts, caches, or generated results.
Changing those policies is deliberately outside this step.

## Scope and workflow groups

Confidence labels mean:

- **High**: directly supported by imports, literal paths, contracts, or output
  manifests.
- **Medium**: purpose is clear, but ownership or current operational status is
  not established.
- **Low / unknown**: insufficient evidence; an owner decision is needed.

### Corrected BTC price-only baseline — active (high confidence)

Located in `old-work/`:

| Path | Observed purpose |
| --- | --- |
| `BASELINE_REBUILD_CONTRACT.md` | Frozen timing, target, execution, cost, and split contract. |
| `baseline_rebuild.py` | Price-only feature/target construction and alignment utilities. |
| `run_baseline_rebuild.py` | Offline one-shot training/evaluation entry point. |
| `test_baseline_rebuild.py` | Alignment, causality, and contract tests. |
| `baseline_rebuild_results/` | Committed summary/config plus generated model, scaler, predictions, and ledgers according to its manifest. |
| `BASELINE_REBUILD_RESULTS.md` | Human-readable validation and reused-test report. |
| `BTC_DATA_INVENTORY.md` | Source coverage, gaps, timestamp interpretation, and partitions. |

The baseline is intentionally offline. The documented command is run from
`old-work/` in conda environment `ml-2`; it reads one-minute BTC market data
from `../db/` and does not import the live bot modules.

### Legacy BTC model audit and reproduction (high confidence)

Located mainly in `old-work/`:

| Path/group | Observed purpose |
| --- | --- |
| `BTC_MODEL_AUDIT.md` | Evidence-backed inventory and compatibility findings for plausible BTC artifacts. |
| `offline_btc_evaluation.py`, `offline_evaluation/` | Offline legacy evaluation code/outputs. |
| `linear-regression*.ipynb` | Legacy feature/target construction and training investigations. |
| `random_forest_model.joblib`, scaler files, `*.pkl`, `*.h5` | Saved models/preprocessors; compatibility must follow the audit rather than filename matching. |
| `bot*.py`, `bin-check.py`, `RealTimeFeatures.py` | Historical/live-oriented execution or feature scripts. These must not be imported merely to evaluate models. |
| `requirements-py3-11.txt` | Old-work dependency snapshot; it differs from the root dependency file. |

The audit records that the original training database is unavailable and that
the current `BatchFeatures.py` interface is not a drop-in reconstruction of the
deployed random-forest preprocessing. It also records mismatched LSTM/scaler
pairs. These are preserved facts, not invitations to silently substitute files.

### Shared market-data and feature utilities (medium confidence)

| Path | Observed purpose / caveat |
| --- | --- |
| `BatchFeatures.py` | Batch feature implementation referenced by research and legacy code; compatibility varies by artifact. |
| `BinanceClient.py` | Exchange/network client used by live or data-acquisition paths. Not needed by the corrected offline baseline. |
| `helper_functions.py` | Large shared helper collection used by root research notebooks; ownership and canonical API are not established. |
| `db/` | Primary observed collection of symbol/timeframe SQLite databases. Large and partly tracked. |
| `BTCUSDT_5m_52weeks.db` | Root-level untracked database; relationship to `db/` copies has not been proven. |
| `logs/` | Runtime/research logs; producers and retention requirements are not fully identified. |

There are 127 database files in `db/`, one database at repository root, and four
database files under `old-work/` at inspection. Apparent duplicates must be
compared by content/schema and consumer before any consolidation.

### Relative-value research (medium confidence)

Root notebooks named `rv-*.ipynb` appear to perform relative-value research
over symbol databases. They use relative data paths and some contain fallback
data-download paths. They are research notebooks, not safe offline entry points
unless network behavior is explicitly disabled.

### RVOL/compression research (medium confidence)

Root notebooks named `rvol-*.ipynb` and `rvol-compression*.ipynb` appear to
explore relative volume and compression signals. They depend on root helpers and
database paths. At least one notebook uses inconsistent literals (`dbs` versus
`./db`), so its working-directory contract is unresolved.

### Unclassified root research (low confidence)

`Notes.ipynb`, `nushur.ipynb`, and `bini-new-exp.ipynb` cannot be assigned to an
active workflow confidently from static inspection alone. `nushur.ipynb` also
had a pre-existing worktree modification. They remain unclassified pending an
owner decision.

### Sentiment work — separate scope (high confidence)

`btc-sentiment-free/`, `btc-sentiment-scorer-v1/`, related evidence/archives,
and their caches form a distinct sentiment project. They are not inputs to the
price-only baseline. The currently identified 2025 sentiment data does not
overlap the corrected baseline's validation/test periods, and no integration is
proposed here.

### Environment and generated state

- Root `requirements.txt`: Python 3.10-era dependency snapshot, including a
  newer scikit-learn than the active baseline environment.
- `old-work/requirements-py3-11.txt`: Python 3.11-oriented legacy snapshot.
- `.venv/`: checked-in Python 3.10.12 virtual environment (large, generated
  state; retention/migration requires a separate decision).
- Conda environments observed include `binbot`, `ml-2`, and sentiment-specific
  environments. The corrected baseline was verified under `ml-2` with
  scikit-learn 1.6.0.
- Notebook kernels vary (`binbot`, `python3`, `ml`, and `ml-2`) and include
  multiple Python minor versions. A notebook kernel name is evidence of author
  intent, not proof that the environment remains reproducible.

## Dependency traces

### Corrected offline baseline

```text
old-work/run_baseline_rebuild.py
  -> old-work/baseline_rebuild.py
  -> old-work/BASELINE_REBUILD_CONTRACT.md
  -> db/BTCUSDT_1m_52weeks.db (relative path: ../db/... from old-work)
  -> training-only StandardScaler + Ridge(alpha=1)
  -> old-work/baseline_rebuild_results/{artifacts,predictions,trades,reports}
```

The process is invoked from `old-work/`; changing the current directory can
change resolution of contract, data, and output paths. The model uses its fixed
feature order and a future executable-return label. Validation and the reused
test partition are reported separately.

### Legacy notebook/model path

```text
old-work/linear-regression*.ipynb
  -> legacy SQLite inputs and notebook-defined preprocessing/features
  -> scaler/model artifacts in old-work/
  -> offline_btc_evaluation.py or historical bot*.py consumers
```

The exact pairings and unresolved incompatibilities are in
`old-work/BTC_MODEL_AUDIT.md`. The historical bot scripts import
`BatchFeatures`/`BinanceClient` from the parent repository while opening model
artifacts by relative path, creating both `PYTHONPATH` and current-directory
assumptions. They may connect to an exchange or submit orders and are not
evaluation commands.

### Root research paths

```text
rv-*.ipynb / rvol-*.ipynb / rvol-compression*.ipynb
  -> helper_functions.py and/or BatchFeatures.py
  -> db/*.db (or inconsistent notebook-relative variants)
  -> notebook-local analysis and plots
```

Some notebooks can acquire missing data through `BinanceClient.py`. Static
inspection alone does not establish a network-free execution contract.

## Paths and operational references

- Relative paths dominate. Root notebooks generally assume the Git root as
  current directory; the corrected baseline assumes `old-work/`; legacy bots
  can require imports from the root while model paths resolve from `old-work/`.
- Hard-coded database-directory variants include `db`, `./db`, and `dbs`.
- Static inspection of accessible system and user service/cron locations found
  no launcher referencing this repository.
- The user crontab contains one job for the separate `psx/stock-watcher`
  project. It uses the `binbot` conda environment name but is not a binbot
  repository launcher.
- No credential contents were printed or copied into this map. Environment and
  secret files require credential-aware handling in any future migration.

## Storage observations

Approximate inspected sizes were: `db/` 989 MiB, `.venv/` 560 MiB,
`btc-sentiment-free/` 1.1 GiB, and `old-work/` 230 MiB. These figures identify
storage-policy pressure only; they are not evidence that files are redundant or
safe to remove.

The authoritative artifact registry for the active baseline should be its
manifest and hashes, not naming conventions. Legacy artifacts require an
explicit compatibility/status manifest before relocation. Large datasets should
be referenced once, never copied merely to satisfy a new layout.
