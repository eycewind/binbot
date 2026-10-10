# Audit of the frozen v3 evaluation

This is a reporting-only audit of the committed selection, references, and model outputs. It does not change the prompt, reference labels, or inference records, and no inference or rescore was performed.

## Ambiguous references and metric denominators

The six ambiguous references were not removed from the originally reported metrics. For each dimension, a model value counted as agreement when it matched that dimension in **any** pre-registered acceptable tuple. Consequently, dimension-wise agreement does not necessarily mean the model matched one complete acceptable tuple: row 4 matches an accepted label and an accepted horizon, but not in the same tuple.

| Row | Pre-registered acceptable tuples | Frozen v3 | Inclusion | Label set | Scope set | Horizon set | Complete tuple |
|---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 4 | `btc_general/mixed/both`; `btc_general/positive/forward` | `btc_general/mixed/forward` | yes | yes | yes | yes | no |
| 11 | `usd_explicit/mixed/current_or_past`; `usd_explicit/neutral/current_or_past` | `usd_explicit/neutral/current_or_past` | yes | yes | yes | yes | yes |
| 17 | `usd_explicit/positive/current_or_past`; `usd_explicit/neutral/current_or_past` | `btc_general/neutral/current_or_past` | yes | yes | no | yes | no |
| 21 | `btc_general/negative/current_or_past`; `btc_general/mixed/both` | `btc_general/mixed/forward` | yes | yes | yes | no | no |
| 24 | `usd_explicit/neutral/current_or_past`; `usd_explicit/negative/current_or_past` | `usd_explicit/neutral/current_or_past` | yes | yes | yes | yes | yes |
| 42 | `btc_general/negative/current_or_past`; `btc_general/neutral/current_or_past` | `btc_general/negative/forward` | yes | yes | yes | no | no |

Their contributions to the original metrics were: structural 6/6, inclusion 6/6, acceptable-label-set 6/6, scope-set 5/6, horizon-set 4/6, and complete acceptable tuple 2/6.

Unambiguous-only results, with the ambiguous six removed from both numerators and denominators, are:

| Metric | Unambiguous only | All references | Ambiguous subset |
|---|---:|---:|---:|
| Structural validity | **42/42** | 48/48 | 6/6 |
| Inclusion/exclusion | **39/42** | 45/48 | 6/6 |
| Sentiment label on reference-included articles | **33/37** | 39/43 | 6/6 |
| Exact scope subtype | **37/42** | 42/48 | 5/6 |
| Horizon | **32/42** | 36/48 | 4/6 |
| Complete scope/label/horizon tuple | **26/42** | 28/48 | 2/6 |

The 39/43 all-reference sentiment result is specifically agreement with the pre-registered acceptable **label sets**. It decomposes into 33/37 unambiguous single-label agreement plus 6/6 ambiguous acceptable-label-set agreement. Complete-tuple agreement is reported separately because independently matching fields can overstate joint agreement.

## Reference-excluded cases

The reference set excludes five cases. The model agreed on two and falsely included three:

| Row | Article | Reference | Exclusion reason | Frozen v3 | Audit |
|---:|---|---|---|---|---|
| 1 | MicroStrategy tax obligation | `unrelated/uncertain/forward` | The possible tax obligation concerns MicroStrategy; no separate BTC claim is made. | `unrelated/uncertain/forward` | Correctly excluded |
| 29 | Solana futures debut comparison | `relative_only/uncertain/current_or_past` | The only BTC claim compares Solana demand with BTC's earlier futures debut. | `btc_general/neutral/current_or_past` | **False inclusion** |
| 30 | Fiscal policy and crypto nerves | `uncertain/uncertain/forward` | A question plus generic “crypto market nerves” does not establish a decisive BTC claim. | `btc_general/positive/forward` | **False inclusion** |
| 38 | Coinone withdrawal court dispute | `unrelated/uncertain/current_or_past` | The claim concerns an exchange, customers, and a court ruling—not BTC direction, demand, or adoption. | `btc_general/neutral/current_or_past` | **False inclusion** |
| 39 | BTC/JPY pair movement | `relative_only/uncertain/current_or_past` | BTC is expressed only through relative JPY-pair movement, without an independent absolute or USD claim. | `relative_only/uncertain/current_or_past` | Correctly excluded |

This confirms the previously reported false inclusions are rows 29, 30, and 38. Rows 29 and 38 add neutral observations to the denominator and can dilute that day's score; row 30 adds both +1 to the numerator and 1 to the denominator.

## Targeted-regression daily impact

The actual `btc_sentiment.daily` branch first puts `uncertain` scope in the uncertain bucket, then puts any scope other than `usd_explicit` or `btc_general` in the excluded bucket. Only an included scope reaches `counts[label]`. Its denominator is `positive + negative + neutral + mixed`, and its numerator is `positive - negative`.

Using that implementation, not merely comparing scope names:

| Failed regression | Reference aggregation | Model aggregation | Daily impact |
|---|---|---|---|
| `target_02_analogy` | `unrelated/uncertain` enters `excluded`: numerator 0, denominator 0 | `relative_only/uncertain` enters `excluded`: numerator 0, denominator 0 | None |
| `target_03_steady_usd` | `usd_explicit/neutral` enters `neutral`: numerator 0, denominator 1 | `btc_general/neutral` enters `neutral`: numerator 0, denominator 1 | None |

Thus the no-impact claim is correct under the implemented aggregation logic. This comparison concerns per-record contribution before exact-headline deduplication; both sides would also be treated identically by that earlier deduplication step.

## Daily contributions for every substantive disagreement

Contribution is shown as `(numerator, denominator)`. A set is shown where the pre-registered ambiguity permits different contributions. These are per-record effects; the resulting change in the percentage depends on the other unique headlines on the same UTC day.

| Row | Disagreement | Reference classification and contribution | Frozen-v3 classification and contribution | Aggregation consequence |
|---:|---|---|---|---|
| 2 | Horizon | `btc_general/positive/both` → `(+1,1)` | `btc_general/positive/forward` → `(+1,1)` | None |
| 6 | Horizon | `btc_general/positive/current_or_past` → `(+1,1)` | `btc_general/positive/forward` → `(+1,1)` | None |
| 8 | Horizon | `usd_explicit/positive/both` → `(+1,1)` | `usd_explicit/positive/forward` → `(+1,1)` | None |
| 9 | Horizon | `btc_general/positive/current_or_past` → `(+1,1)` | `btc_general/positive/forward` → `(+1,1)` | None |
| 13 | Label, horizon | `usd_explicit/mixed/current_or_past` → `(0,1)` | `usd_explicit/negative/forward` → `(-1,1)` | Numerator −1 |
| 15 | Horizon | `btc_general/mixed/current_or_past` → `(0,1)` | `btc_general/mixed/both` → `(0,1)` | None |
| 17 | Scope | accepted positive or neutral → `{(+1,1),(0,1)}` | `btc_general/neutral/current_or_past` → `(0,1)` | None under the accepted neutral reference; numerator −1 versus the accepted positive reading |
| 18 | Label, horizon | `btc_general/neutral/current_or_past` → `(0,1)` | `btc_general/positive/forward` → `(+1,1)` | Numerator +1 |
| 21 | Horizon | accepted negative or mixed → `{(-1,1),(0,1)}` | `btc_general/mixed/forward` → `(0,1)` | None under the accepted mixed reference; numerator +1 versus the accepted negative reading |
| 25 | Horizon | `usd_explicit/positive/current_or_past` → `(+1,1)` | `usd_explicit/positive/forward` → `(+1,1)` | None |
| 27 | Horizon | `btc_general/positive/both` → `(+1,1)` | `btc_general/positive/forward` → `(+1,1)` | None |
| 29 | False inclusion, scope | `relative_only/uncertain/current_or_past` → `(0,0)` | `btc_general/neutral/current_or_past` → `(0,1)` | Denominator +1; neutral dilution |
| 30 | False inclusion, scope | `uncertain/uncertain/forward` → `(0,0)` | `btc_general/positive/forward` → `(+1,1)` | Numerator +1 and denominator +1 |
| 35 | Label, scope | `usd_explicit/positive/current_or_past` → `(+1,1)` | `btc_general/neutral/current_or_past` → `(0,1)` | Numerator −1 |
| 36 | Horizon | `usd_explicit/positive/current_or_past` → `(+1,1)` | `usd_explicit/positive/forward` → `(+1,1)` | None |
| 38 | False inclusion, scope | `unrelated/uncertain/current_or_past` → `(0,0)` | `btc_general/neutral/current_or_past` → `(0,1)` | Denominator +1; neutral dilution |
| 41 | Label | `btc_general/neutral/forward` → `(0,1)` | `btc_general/positive/forward` → `(+1,1)` | Numerator +1 |
| 42 | Horizon | accepted negative or neutral → `{(-1,1),(0,1)}` | `btc_general/negative/forward` → `(-1,1)` | None under the accepted negative reference; numerator −1 versus the accepted neutral reading |
| 43 | Scope | `usd_explicit/negative/current_or_past` → `(-1,1)` | `btc_general/negative/current_or_past` → `(-1,1)` | None |

## Offline test-suite audit

The seven-test run used `python -m unittest -v test_score_v2 test_score_v3`, explicitly restricting execution to the six v2 validator tests and one v3 prompt-patch test. It was not complete test discovery.

`python -m unittest discover -v` under the `binbot` environment discovers six applicable modules:

| Module | Tests |
|---|---:|
| `test_collection.py` | 4 |
| `test_pipeline.py` | 4 |
| `test_revalidation.py` | 2 |
| `test_score_v2.py` | 6 |
| `test_score_v3.py` | 1 |
| `test_scoring.py` | 2 |
| **Total** | **19** |

The complete applicable offline suite passed: **19/19**. Some mocked collection/scoring tests print progress and failure-path diagnostics by design; they did not contact the model or perform live inference.
