# Manual review of `sample_v2_results.jsonl`

Reviewed against `CLASSIFICATION_CONTRACT_V3.md`. `OK` means the complete scope/label/horizon decision is supported. Evidence selection and reason were also checked against the supplied H/S text. Ambiguous rows are not counted as forced errors.

| # | Manual decision | Review of v2 |
|---:|---|---|
| 1 | `unrelated/uncertain/current_or_past` | Error: a Litecoin/“Bitcoin's gold” analogy is not BTC relative performance. |
| 2 | `btc_general/neutral/current_or_past` | OK; evidence supports a transfer into BTC, not directional sentiment. |
| 3 | `btc_general/negative/current_or_past` | Horizon error: the reserve rejection is a completed/current decision, not a forecast. |
| 4 | `usd_explicit/positive/forward` | OK. |
| 5 | `usd_explicit/mixed/forward` | Ambiguous but likely error: “strong demand” is positive while constrained gains/fleeting rally is negative; the contract retains both. |
| 6 | `btc_general/positive/current_or_past` | Horizon/reason error: “Bitcoin climbs” is the supported positive signal; inferred future adoption is not. |
| 7 | `usd_explicit/mixed/forward` | OK; H and S preserve both directions. |
| 8 | `btc_general/positive/current_or_past` | Horizon error: the launch already occurred. Positive adoption/utility is explicit enough under the contract. |
| 9 | `btc_general/positive/forward` | Ambiguous: investment is in a company but is explicitly intended to support future BTC accumulation. Acceptable positive reading. |
| 10 | `usd_explicit/positive/current_or_past` | OK. |
| 11 | `usd_explicit/negative/forward` | OK. |
| 12 | `btc_general/negative/current_or_past` | Ambiguous: “bleeds” is a negative BTC cue and both named ETFs moved negative; acceptable, though the summary gives no per-asset magnitude. |
| 13 | `btc_general/positive/forward` | OK. |
| 14 | `btc_general/positive/current_or_past` | Scope error: an absolute rise without a USD denomination is `btc_general`, not `usd_explicit`. |
| 15 | `usd_explicit/positive/forward` | OK; “cautious ... hope” still expresses a qualified bullish target. |
| 16 | `btc_general/positive/forward` | Ambiguous: this is advocacy for adoption, not a completed adoption; acceptable positive signal. |
| 17 | `btc_general/positive/current_or_past` | OK; the per-BTC purchase price is transaction detail, not necessarily a market quote. |
| 18 | `btc_general/positive/forward` | Scope error: “bullish cues” is an absolute BTC outlook but no BTC/USD quote is supplied. |
| 19 | `usd_explicit/positive/current_or_past` | OK. |
| 20 | `relative_only/uncertain/forward` | OK. |
| 21 | `relative_only/uncertain/unspecified` | Ambiguous: H establishes a comparison topic, but S states only gold information; `uncertain` scope is also defensible. |
| 22 | `btc_general/negative/current_or_past` | OK; S independently states a BTC-price slide. |
| 23 | `btc_general/negative/current_or_past` | Error: BTC explicitly dropped 6–10%; listing other coins does not make an absolute fall relative-only. Horizon is not forward. |
| 24 | `usd_explicit/neutral/current_or_past` | OK; generic S is irrelevant but selected H supports the result. |

Strictly reviewable rows: 20 (excluding 5, 9, 12, 21). Of those, 13 are fully correct and 7 have a scope, label, horizon, or rationale error. Scope/label alone is correct on 16/20; errors are rows 1, 14, 18, and 23. This is a diagnostic sample used for tuning, not an accuracy estimate.
