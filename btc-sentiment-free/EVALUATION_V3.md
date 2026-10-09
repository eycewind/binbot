# Classifier v3 limited evaluation

The v3 change is prompt-only. It leaves the schema, validator, sampling, database, and prior JSONL files unchanged. Both evaluations used local `qwen3.5:9b`, temperature 0, seed 42, 4096 context, and 1024 output tokens.

## Synthetic benchmark

| Measure | v2 | v3 |
|---|---:|---:|
| Structurally valid | 20/20 | 20/20 |
| Exact original expected scope/label | 12/20 | 16/20 |
| Exact after justified case-17 correction | 13/20 | 17/20 |

Fixed: cases 11, 12, 16, and 18. Remaining semantic errors: 07 (passing mention called relative), 10 (explicit unchanged USD price called general), and 13 (unattributed combined flows called positive). Case 17 now disagrees with the unchanged fixture but agrees with the separately documented corrected gold.

## Real 24-item diagnostic sample

Four ambiguous rows (5, 9, 12, 21) are not forced into the denominator.

| Measure on 20 reviewable rows | v2 | v3 |
|---|---:|---:|
| Correct scope and label | 16/20 | 18/20 |
| Correct scope, label, and horizon | 13/20 | 15/20 |

Improvements include row 3 horizon, row 14 scope, row 18 scope, and row 23 scope/label. Remaining errors include row 1 scope; rows 6 and 8 horizons; row 15 scope (a new regression: `$100K` should be `usd_explicit`); and row 23 horizon. Row 5 improved to `mixed` but its `btc_general` scope conflicts with the `$110K` price target and the row remains intrinsically borderline because “strong demand” is a concessive clause.

These are tuned diagnostic sets, not held-out accuracy estimates.
