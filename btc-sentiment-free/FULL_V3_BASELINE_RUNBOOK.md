# Full-dataset frozen-v3 baseline runbook

This workflow produces a classification baseline only. It does not collect articles, change the frozen v3 classifier, train a trading model, or evaluate profitability.

## Frozen identity

- Articles: 809 (the expected count), spanning both publishers in all 12 months of 2025.
- Input-set hash: `4e6afe42fbe3771ab1c8629ea01fa8fe8207914bbecc598ac891e7fff5ec0dc6`.
- Manifest SHA-256: `8bcf51d4960b0878e5c087880a9a90296a8b34fee1d2d9f4f77687c7bb7f853c`.
- Frozen SQLite backup SHA-256: `f2fcc74488d99382f497123c3c634039bb2851ced6189a7b7dc9ae3731cb9761`.
- Baseline run ID: `7c2ad3400408cce53c1d0db4cc687350773b43058144d9163a8bbc1cf9701672`.
- Model: local `qwen3.5:9b`, digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
- Options: temperature 0, seed 42, context 4096, output budget 1024, thinking off.
- Maximum attempts: one per article. A failed generation is a final failed outcome and is not silently retried.

The committed manifest contains each exact headline and summary, stored collection text hash, independently recomputed input hash, source, URL, and published/modified/collected timestamps. `FULL_V3_FREEZE.json` records source/month coverage. The original database and all existing runs remain unchanged.

## Runtime state and provenance

Large/runtime artifacts are intentionally ignored under `full_v3_runtime/`:

- `frozen.sqlite`: SQLite online-backup snapshot of the original database.
- `results.sqlite`: raw requests/responses, failures, outcomes, provenance, and strict run/input identity.
- `articles.csv`: final article reconciliation with timestamps and nominal next-day availability.
- `daily.csv`: 365-day pooled and per-publisher export.

The initialized runtime state contains 72 verified successes reused from `sample_v3_results.jsonl` (24) and `validation_v3_results.jsonl` (48). Reuse required exact prompt, schema, model digest, settings, title, summary, input hash, raw-response replay, and validator agreement. No v1/v2 scores were imported. The remaining 737 inputs require one new local inference each.

## Reproducible commands

From this directory:

```bash
conda run -n binbot python --version
conda run -n binbot python -m unittest discover -v

# Already completed for the current runtime state; safe and idempotent checks:
conda run -n binbot python full_v3_baseline.py init --identity-jsonl validation_v3_results.jsonl
conda run -n binbot python full_v3_baseline.py import-existing sample_v3_results.jsonl validation_v3_results.jsonl

# Resumes only unattempted exact inputs and re-verifies the live model identity first:
conda run -n binbot python full_v3_baseline.py score

# Refuses to export until all 809 inputs have a final success/failure outcome:
conda run -n binbot python full_v3_baseline.py export
```

The daily score is `100 * (positive - negative) / (positive + negative + neutral + mixed)`. No usable observations produce a blank score, never zero. Exact normalized headlines are deduplicated within UTC day before aggregation. Excluded, uncertain, and failed/unscored counts remain separate. Per-publisher counts/scores use the same pooled deduplication decisions.

`nominal_next_day_utc` is a research assumption for later lagged joins. It is not verified historical availability. These retrospectively collected current texts are not point-in-time safe.
