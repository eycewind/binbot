# Offline evidence revalidation

Add revalidate_evidence.py beside your existing scorer-v1 btc_sentiment.py. Do not replace the scorer, database or cache. Python standard library only; use your existing binbot environment. Stop any scoring process before running revalidation; back up the SQLite database first. No model calls or downloads occur.

Run:

    python revalidate_evidence.py --source-run 816bd91eedaf5d431b76cdc08483cc60e9ce48acc6b535471a4d2d2373e05897

Copy REVALIDATED_RUN_ID from the output, then export:

    python btc_sentiment.py daily --run REVALIDATED_RUN_ID --output daily_2025_qwen3p5_revalidated.csv

The source scores, run and attempts are not changed. A deterministic derived run holds copied valid scores and newly recovered evidence matches. Rerunning revalidation after more source scoring updates that same derived run. Continue inference with the original score command/model/settings, not the derived run ID. The derived CSV changes only when you re-export it.

Only curly/straight single/double quotes, character case and whitespace differences are normalized. No punctuation removal, fuzzy matching, ellipsis expansion, paraphrase matching or quote-wrapper stripping. Every accepted match is mapped back to the exact contiguous original source span. Stored score evidence is that source span; evidence_audit preserves the model's original string, source offsets, match rule, source run and attempt ID. Truncated/empty/malformed generations still fail. Only the most recent attempt for each failed article is considered; the tool does not cherry-pick older outputs.

Schema and evidence validation do not prove sentiment accuracy. Labels and explanations are unchanged. Crime-related negative wording does not automatically imply negative BTC/USD sentiment; questions about higher prices are not by themselves bullish predictions. Review these distinctions separately before interpreting the daily scores.

Validation: replaying the supplied 20 failures recovered 16 formatting-only evidence matches and rejected 4 altered/noncontiguous quotations. The export contains the previous seven failures, so 20 is not necessarily a new-batch failure count or full run total. Two additional tests passed for normalization boundaries, exact-source provenance, repeated replay, preserved source scores and unattempted records. No live inference was performed or needed.
