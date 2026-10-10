# BTC/USD daily news sentiment — free research prototype

Target: Bitcoin's USD outlook, 2025-01-01 through 2025-12-31 inclusive, UTC.
No API subscriptions, API keys, or paid model calls. Python 3.10+ standard library only.
This is an acquisition/scoring prototype, not a completed 2025 dataset or validated trading signal.

## What it does

1. `discover`: read public publisher sitemaps (CoinDesk and Bitcoin.com News), record candidate URLs.
2. `collect`: retrieve allowed public pages, extract publication time, modification time, headline and publisher summary. Save a local SQLite database and cached HTML.
3. `score`: use an already installed local Ollama model to classify BTC/USD relevance and sentiment. Record model digest and prompt hash. Reject malformed output and invented evidence.
4. `daily`: export 365 rows, keeping missing observations distinct from neutral sentiment. No plotting or price analysis is mixed into scoring.

Both publishers belong to the crypto industry; separate publisher names are not proof of independent reporting. Deduplication in v0 removes exact normalized headline copies only. Paraphrased/syndicated versions still require review or a later event-clustering step.

## Run on your Ubuntu machine

Use an isolated environment; this does not modify the Binance repository or its environment.

```bash
conda create -n btc-sentiment python=3.11 -y
conda activate btc-sentiment
python --version
# Change directory to the extracted btc-sentiment-free folder.
python -m unittest -v
python btc_sentiment.py discover
python btc_sentiment.py collect --limit 30
ollama list
# Replace EXACT_MODEL_NAME with an already installed model name from ollama list.
python btc_sentiment.py score --model EXACT_MODEL_NAME --limit 30
# Copy the RUN_ID printed by score:
python btc_sentiment.py daily --run RUN_ID --output daily_2025.csv
```

Ollama must already be serving on localhost:11434. The script does not download models or contact a cloud inference service. A small local model is sufficient to test the workflow, but its classification quality must be reviewed. Temperature 0 and a fixed seed reduce variability; they do not promise bitwise repeatability across runtimes/hardware.

Start with 30 collected articles and review the results before requesting thousands. Repeat `collect` and `score` to advance the batch; successful articles and run-specific scores are resumable. Errors are recorded in the `errors` table and remain eligible for retry. A repeatedly failing early batch can block progress: inspect errors and fix access/extraction issues before expanding. Redirects are intentionally not followed; update publisher URL adapters explicitly when necessary.

Discovery may scan many archive pages, particularly for Bitcoin.com. `discover --sources coindesk --max-sitemaps 4` is a small connectivity test only, NOT a representative sample or complete-year discovery. A sitemap's last-modified date is never treated as publication time; it only prioritizes collection. Cached sitemap snapshots are frozen for reproducibility. Use a new `--cache` directory if refreshing discovery is intended.

Global options go before the command:

```bash
python btc_sentiment.py --db experiment.sqlite --cache experiment-cache --start 2025-01-01 --end 2025-12-31 discover
```

## Inspect evidence

Use Python's built-in sqlite3 to inspect each record; no database client is required:

```bash
python - <<'PY'
import sqlite3, json
c = sqlite3.connect('btc_sentiment.sqlite')
for title, url, result in c.execute('SELECT a.title,a.url,s.result FROM articles a JOIN scores s ON a.url=s.url LIMIT 30'):
    print(title, url, json.loads(result), sep='\n')
    print()
PY
```

For serious review filter to the exact run ID; the query above can include multiple model runs.

## Locked interpretation rules

- Include explicit BTC/USD statements and substantive BTC-specific developments without a currency quote.
- Exclude pure ETH/BTC or BTC/ETH relative-performance claims. BTC can outperform ETH while still declining in USD.
- An ETH mention alone does not disqualify an article. Score only the BTC-relevant statement.
- Bitcoin Cash and Bitcoin SV are not Bitcoin. V0 excludes their explicit URL slugs; ambiguous cases still need model/manual review.
- Retain uncertain and mixed classifications rather than forcing positive/negative.
- Classify sentiment expressed in the supplied text, not a predicted trade outcome. Do not invent market reactions from generic topics.

## Daily score

After exact-headline deduplication within a UTC day:

`100 * (positive_count - negative_count) / (positive + negative + neutral + mixed)`

Positive contributes +1, negative -1, neutral and mixed 0. Mixed has its own count and is not semantically neutral. Uncertain, unrelated and relative-only classifications are excluded from the denominator. No usable observations yields an empty score, not zero. Publisher counts measure publisher coverage, not independence. All usable scores are labeled provisional; incomplete scoring is flagged separately. They do NOT prove that collection for that day is complete.

## Known limitations — do not skip these

- Only headlines and publisher summaries are scored. Full-article sentiment and passage extraction are not implemented.
- URL discovery requires 'bitcoin' or 'btc' in the URL path. Articles with other slugs, missing sitemap entries, removed pages and inaccessible sources are missed. This is a documented recall limitation.
- Full-year and per-month coverage have not been established. The pilot verifies individual page extraction, not archive completeness. Neither current RSS feeds nor web-search snippets are a substitute for a full historical archive.
- Multiple publishers may repeat one underlying story. Exact-headline deduplication does not solve event duplication or publisher-volume bias.
- Publication dates come from current page metadata. Current text may have changed since 2025. Store modification and collection times, and do not call the result point-in-time-safe.
- An LLM trained after 2025 can have hindsight in its parameters. Instructions to use only evidence mitigate, but do not eliminate, this issue.
- Retrospective article selection has survivorship bias; article volume can itself vary with market activity and collection failures.
- BTCUSDT is a price-comparison proxy for BTC/USD, not an identical instrument. No price data is downloaded in this prototype.
- No claims of trading performance. A day-D final score must not be used before day D closes. Actual availability timing needs separate verification.

## Next acceptance gate

Before scaling, manually label a date-stratified sample across January, middle months and December, covering both sources and the cross-pair edge cases. Check relevance errors separately from sentiment errors. Report per-month article counts, contributing publishers, failures, uncertain labels and duplicate stories. Expand URL discovery and event deduplication before interpreting a full-year chart. Test forward returns only after this data-quality gate; same-day price/sentiment correlation is not predictive evidence.

## Verification

`python -m unittest -v` checks UTC day boundaries, publication-vs-modification handling, robots rules, evidence validation, relative-only exclusion, duplicate-headline aggregation and missing-day handling. Synthetic tests verify mechanics, NOT language-model classification accuracy. See `pilot_check.json` for actual live extraction checks. Local Ollama scoring was not executed in the build workspace.

## References checked during implementation

- https://www.coindesk.com/robots.txt
- https://www.coindesk.com/sitemap-index.xml
- https://news.bitcoin.com/robots.txt
- https://news.bitcoin.com/sitemap.xml
- https://docs.ollama.com/api/generate

Access policies and archive layouts can change. The collector checks robots policies on each invocation, rate limits requests and records access errors. It does not bypass authentication, paywalls, CAPTCHAs or other access controls.


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


# Collection fix 1

Put collect_balanced.py alongside your existing btc_sentiment.py. No replacement
of that file is needed. Python standard library only; use conda activate binbot.
Stop any existing collector before running this companion.

Commands from the project directory:

    python collect_balanced.py coverage
    python collect_balanced.py plan --limit 100
    python collect_balanced.py collect --limit 100

Use --db PATH and --cache PATH if your existing project uses nondefault locations.
Repeat collect to append new articles. Existing articles and scores are preserved.
Use this collector instead of the old `btc_sentiment.py collect` command.
--limit counts attempted URLs, not successful articles.

Scheduling rotates sources and month-hint buckets, retaining attempt counts across
runs for the specified date range. SHA-256 URL ordering removes alphabetical
ordering within each bucket. Out-of-range/missing hints get an unknown bucket;
only extracted publication timestamps determine inclusion. Previously collected
articles are not rebalanced or removed. Fair scheduling cannot guarantee balanced
successful downloads, genuine publication months, or representative coverage.

Failed URLs are skipped by default (including old article errors). To retry them:

    python collect_balanced.py collect --limit 100 --retry-failed

Fresh URLs within a source precede retries. Failures and out-of-period outcomes are
recorded in collection_attempts_v2. Existing visited records are not used because
they do not record the date range; a small amount of old out-of-period work may be
revisited once. Do not run multiple collectors concurrently.

If a source has zero candidates, run discovery with the existing script (see
`python btc_sentiment.py discover --help`) and inspect its source errors. This
patch does not change sitemap discovery or URL-keyword screening. Publisher access
has not been tested live in this delivery. Empty months remain missing, not neutral.

Do not rescore the expanded dataset yet: classification fixes are the next step.

Tests, after placing files alongside btc_sentiment.py:

    python -m unittest test_collection -v

# BTC classifier v2 pilot

Extract these files beside your existing btc_sentiment.py. Activate binbot.
No added dependencies. Ollama must be running locally. The database is opened
read-only for sampling; benchmark mode does not open it. Existing scores and
run IDs remain unchanged. Outputs are new JSONL files; existing files are refused.

1. Run the 20 synthetic regression cases:

    python score_v2.py benchmark --model qwen3.5:9b --output benchmark_v2_results.jsonl

2. Run the real-data pilot (12 per source, spread across available months):

    python score_v2.py sample --model qwen3.5:9b --per-source 12 --output sample_v2_results.jsonl

Upload both JSONL files for review. If inference fails extensively, stop with Ctrl-C;
completed lines are flushed. A repeat requires a different output filename and
reruns the pilot; this small pilot does not implement resume.

The 20 synthetic cases have author-assigned expected scope/label values. They
are regression checks designed alongside the prompt, NOT independent validation
or a measure of real-world accuracy. Horizons are recorded but not benchmarked.
The real sample is for manual review, not held-out validation once used for tuning.
Do not run a full rescore until this pilot is reviewed. No daily export is included.

Evidence IDs select exact headline H / summary S fields; quotations are retrieved
by Python. This guarantees text provenance, NOT relevance or logical support.
Mixed labels should cite all conflicting evidence. Short summaries can be
insufficient. The target is expressed BTC/USD news sentiment; no return forecasts
are evaluated. Neutral differs from uncertain; excluded scopes use uncertain.

BTC-specific ETF demand/adoption changes can receive a directional label even
without a USD quote. Company-stock moves and relative-only asset comparisons
are excluded unless separate BTC demand or absolute direction is expressed.
This operational definition must remain fixed when comparing runs.

Defaults: temperature 0, seed 42, context 4096, output budget 1024. Thinking is
disabled for models advertising that capability. Metadata records model digest,
prompt, schema, options and input hash. Raw responses are preserved for failures
and successes. Identical settings do not guarantee identical GPU outputs.

Offline tests:

    python -m unittest test_score_v2 -v

Six validator tests passed during preparation. No local model inference was
available in the authoring environment; model quality remains unverified.
