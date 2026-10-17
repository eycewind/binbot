# Scorer v1 update

Replace btc_sentiment.py in your existing working directory. Keep btc_sentiment.sqlite and cache. Back up the database first with Python sqlite3.Connection.backup if another process might be writing. Existing tables/runs are retained; score creates an additive score_attempts table. There are no new dependencies (Python 3.10+).

Use the same existing environment and verify `python --version` and `which python`; no package installation or environment changes are needed. Run `python -m unittest -v` after copying the included test files if desired.

Start with:

    python btc_sentiment.py score --model qwen3.5:9b --limit 5

Default: 4096 context, 1024 output tokens, thinking off when /api/show advertises the thinking capability, JSON schema, exact evidence validation, three-consecutive-failure stop. If the installed Ollama does not advertise thinking, no unsupported thinking flag is sent. Errors remain errors; they are never converted to neutral scores. The exact evidence requirement is retained: schema constrains structure but cannot ensure a semantically correct label or quotation.

The new settings generate a NEW RUN_ID. Use that printed ID, not the old run ID:

    python btc_sentiment.py daily --run NEW_RUN_ID --output daily_2025_qwen3p5_v1.csv

If the five-item smoke test succeeds, continue with the SAME command settings and --limit 50. Successful scores are skipped. New, unattempted articles are prioritized ahead of failed ones, then oldest publication/URL. A new model/configuration starts a separate scoring run over existing articles: no recollection is required. Prior scores remain available with their original run IDs.

If it fails:

    python btc_sentiment.py diagnostics --run NEW_RUN_ID --output score_failures.jsonl

Share the diagnostic file. It contains the supplied public article text, complete request, raw response (including any returned thinking), stop reason, token counts when returned, and validation error. Transport errors have a null response. Diagnostics overwrite the chosen JSONL file. score_attempts preserves the full history in SQLite. Old failed raw responses cannot be recovered retroactively.

Optional explicit settings:

    python btc_sentiment.py score --model gemma4:12b --limit 5 --num-ctx 4096 --num-predict 1024 --think off

Changing context, output budget or effective thinking flag changes run identity. `--think default` omits the control. `--think on` enables it only for models advertising support. No automatic token escalation or silent fallback mixes settings. Raw JSON must parse as one complete object with exactly the expected fields; incomplete outputs are rejected even if partially parseable. Failure scheduling and early stopping prevent endless repeats, but do not guarantee model compliance.

## What is and is not fixed

Implemented schema-based output, explicit budgets, thinking control, raw response retention, truncation/empty-response diagnostics, run isolation, progress counts, early stopping, and scoring queue fairness. Headline collection, publisher coverage, collection retry behavior, historical point-in-time limitations and semantic accuracy remain separate issues. The 2025 corpus is unchanged by this update.

Validation: 6 local tests passed, including mocked Ollama responses for empty/thinking-only output, truncation, raw response persistence, early stopping, prioritization of unattempted articles and preservation of old runs. No live inference was possible on the user's GPU from this workspace. Actual Qwen/Gemma behavior must be confirmed by the five-article run.

References: https://docs.ollama.com/api/generate and https://docs.ollama.com/capabilities/structured-outputs
