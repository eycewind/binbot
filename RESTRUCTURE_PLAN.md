# Staged Repository Reorganization Plan

This is a proposal only. No existing file is moved, renamed, deleted, or
reclassified in Git by this document.

## Proposed layout

```text
binbot/
├── README.md
├── docs/
│   ├── repo-map.md
│   ├── environments.md
│   └── storage-policy.md
├── contracts/
│   └── btc_price/
├── src/
│   └── btc_price_baseline/
├── tests/
│   └── btc_price_baseline/
├── research/
│   ├── relative_value/
│   └── rvol_compression/
├── archive/
│   └── legacy_btc/
│       ├── notebooks/
│       ├── scripts/
│       ├── live_bots_disabled/
│       └── reports/
├── data/
│   └── market/                 # one canonical reference or external mount
├── artifacts/
│   ├── btc_price_baseline/
│   └── legacy_btc/
├── results/
│   └── btc_price_baseline/
└── btc-sentiment-free/         # remains a separate project area
```

`data/`, `artifacts/`, and `results/` describe logical ownership. Their final
physical/Git/LFS/external-storage policy must be agreed before moving large
files. This plan does not duplicate datasets.

## Proposed old-to-new mapping

| Existing path/group | Proposed destination | Condition |
| --- | --- | --- |
| `old-work/baseline_rebuild.py` | `src/btc_price_baseline/data.py` | Preserve public behavior; update imports and hashes deliberately. |
| `old-work/run_baseline_rebuild.py` | `src/btc_price_baseline/evaluate.py` or a thin `scripts/` entry point | Add root-independent path resolution first. |
| `old-work/test_baseline_rebuild.py` | `tests/btc_price_baseline/` | Tests must pass before and after the move. |
| `old-work/BASELINE_REBUILD_CONTRACT.md` | `contracts/btc_price/BASELINE_REBUILD_CONTRACT.md` | Preserve content/hash reference or version the contract explicitly. |
| `old-work/BTC_DATA_INVENTORY.md` | `contracts/btc_price/BTC_DATA_INVENTORY.md` | Keep source hashes and split declarations. |
| `old-work/BASELINE_REBUILD_RESULTS.md` | `results/btc_price_baseline/<run-id>/SUMMARY.md` | Retain the existing run identity; do not rerun to manufacture equivalence. |
| `old-work/baseline_rebuild_results/` | `results/btc_price_baseline/<run-id>/` plus `artifacts/btc_price_baseline/<run-id>/` | Split only with a manifest recording every old/new path and hash. |
| `old-work/BTC_MODEL_AUDIT.md` | `archive/legacy_btc/reports/` | Keep linked from root README. |
| `old-work/offline_btc_evaluation.py` and results | `archive/legacy_btc/scripts/` and `archive/legacy_btc/reports/` | Preserve offline reproducibility; do not import live code. |
| `old-work/linear-regression*.ipynb` | `archive/legacy_btc/notebooks/` | Record kernel and artifact dependencies in a manifest. |
| `old-work/bot*.py` | `archive/legacy_btc/live_bots_disabled/` | Move only after launcher search and explicit safety labeling. |
| Other legacy scripts (`bin-check.py`, `RealTimeFeatures.py`, etc.) | `archive/legacy_btc/scripts/` | Confirm consumers first. |
| Legacy `*.joblib`, `*.pkl`, `*.h5` | `artifacts/legacy_btc/` | Hash and pair with preprocessing/feature metadata; never select by results. |
| `rv-*.ipynb` | `research/relative_value/` | Establish an offline data-access layer before moving. |
| `rvol-*.ipynb`, `rvol-compression*.ipynb` | `research/rvol_compression/` | Resolve `db`/`dbs` ambiguity first. |
| `helper_functions.py` | Initially beside its proven research consumers | Do not promote it to shared library until its API/owners are known. |
| `BatchFeatures.py`, `BinanceClient.py` | Leave in place initially; later archive or package by proven consumers | Network client and artifact-specific feature code should not be conflated. |
| `db/` and other `*.db` | `data/market/` logical namespace or external data root | Inventory schema/content hashes and references; do not copy large files. |
| `logs/` | `archive/legacy_btc/logs/` or external retention | Identify producer and retention obligations first. |
| `Notes.ipynb`, `nushur.ipynb`, `bini-new-exp.ipynb` | `research/unclassified/` only after owner confirmation | Purpose presently unknown. |
| `.venv/` | Recreate from a lock/spec; remove from Git only in a later approved change | Verify no scripts embed its absolute interpreter path. |
| Sentiment directories/archives | No move in this plan | Keep separate until integration is explicitly requested. |

## Storage policy to establish before migration

- **Source and tests:** normal Git tracking, small and reviewable.
- **Contracts/reports:** Git-tracked, immutable per evaluated run; amendments
  receive a new version or explicit change record.
- **Raw market data:** one canonical copy outside ordinary Git where practical,
  referenced by a configurable data root and content/schema manifest.
- **Models/scalers:** immutable run-addressed artifacts with SHA-256, library
  versions, feature order, target units, and training bounds.
- **Predictions/trade ledgers:** run-addressed generated results with input and
  config hashes. Decide Git versus artifact storage based on size; summaries
  stay in Git.
- **Logs/caches/checkpoints:** generated state with a retention policy, never
  treated as source. Do not add ignore rules until current tracked consumers are
  understood.
- **Secrets:** never migrate into documentation or artifacts; retain via an
  explicit secret-management mechanism outside committed files.

## Safe migration batches

### Batch 0 — decisions and manifests

No moves. Resolve the scope questions below; create machine-readable inventories
of data and artifacts containing size, SHA-256, schema/type, known consumer, and
confidence. Record baseline commands and environment exports without installing
anything.

Verification: cleanly distinguish newly generated manifests from pre-existing
worktree changes; compare manifest counts/hashes to the current tree.

### Batch 1 — package the active price-only baseline

Move only baseline source/tests/contracts. Make paths relative to the Git root or
explicit CLI arguments rather than the process current directory. Keep exchange
clients out of the package.

Verification: run the existing tests under verified `ml-2`; run the frozen
evaluation against the same input and compare config/input hashes, prediction
rows, trade ledgers, metrics, and serialized model/scaler metadata. A difference
requires explanation, not silent regeneration.

### Batch 2 — separate active artifacts and results

Create run-addressed artifact/result directories and a compatibility manifest.
Relocate the existing baseline run without retraining.

Verification: pre/post SHA-256 equality, manifest path validation, summary-link
validation, and a load-only inference smoke test in `ml-2`.

### Batch 3 — archive the legacy BTC workflow

Move audited notebooks/scripts/reports and disable-by-location live bot entry
points. Preserve every model/scaler; add metadata rather than deleting apparent
duplicates.

Verification: pre/post hashes, static import/path scan, notebook JSON validity,
artifact load tests only where the audited environment is known, and a repeat
search for external launchers before live-script relocation.

### Batch 4 — organize relative-value and RVOL research

Treat these as two research workflows. Replace notebook-specific path literals
with a read-only configurable data root only after capturing current behavior.

Verification: notebook static validation, dependency/path inventory, and
offline smoke tests that fail closed rather than download missing market data.

### Batch 5 — consolidate data references and logs

After consumers are known, select one canonical storage location for each unique
database and define log retention. Use references/configuration, not copied
datasets.

Verification: database schema/row-range/content checks, pre/post hashes where
byte preservation is expected, consumer path tests, disk-usage comparison, and
explicit rollback mapping.

### Batch 6 — generated-state and tracking policy

Only after prior batches, decide how to handle `.venv/`, caches, checkpoints,
archives, and large results; then propose `.gitignore`, lockfile, or artifact
storage changes for separate approval.

Verification: fresh environment creation from the chosen specification,
dependency/version report, baseline tests, and proof that no required artifact
became unreachable.

## Scope questions before migration

1. Which root notebook, if any, is the current canonical RVOL/relative-value
   entry point, and are either workflows still active?
2. What are the intended purposes/owners of `Notes.ipynb`, `nushur.ipynb`, and
   `bini-new-exp.ipynb`?
3. Are any legacy `old-work/bot*.py` scripts launched outside the inspected
   system/user cron and systemd locations (for example Docker, supervisor, a
   remote host, or a manual runbook)?
4. Should raw databases remain local/external, use Git LFS, or move to a shared
   data root? The answer should precede any path migration.
5. Should large baseline predictions and trade ledgers remain committed, move
   to artifact storage, or use Git LFS while summaries/manifests stay in Git?
6. May the checked-in `.venv/` eventually be replaced by a reproducible conda
   or lockfile specification after embedded-path checks?
7. Should the sentiment project ultimately become a separate repository, or
   remain a clearly isolated top-level area in this repository?
