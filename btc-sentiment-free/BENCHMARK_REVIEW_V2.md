# Review of v2 benchmark mismatches

Baseline: 12/20 exact expected scope/label matches; all 20 responses were structurally valid.

| Case | Expected | v2 | Assessment |
|---|---|---|---|
| 07 | `unrelated/uncertain` | `relative_only/uncertain` | Model error. A mere mention of BTC “among other assets” is not an actual relative BTC claim. |
| 10 | `usd_explicit/neutral` | `btc_general/neutral` | Model error. “Unchanged in USD” explicitly denominates BTC price behavior. |
| 11 | `btc_general/positive` | `usd_explicit/positive` | Model error. `$300 million` describes ETF flows, not BTC's dollar price. |
| 12 | `btc_general/negative` | `usd_explicit/negative` | Same scope error as case 11. |
| 13 | `btc_general/uncertain` | `btc_general/neutral` | Model error. Combined BTC/ETH flows cannot be directionally attributed to BTC; missing attribution is insufficient, not neutral. |
| 16 | `btc_general/mixed` | `relative_only/uncertain` | Model error. Rally and decline are independent, conflicting absolute BTC forecasts; no asset comparison is present. |
| 17 | `btc_general/uncertain` | `uncertain/uncertain` | Gold expectation correction: change scope to `uncertain`. The sentence ends before its predicate, so it establishes neither an absolute nor relative BTC claim. This correction is semantic and would remain justified regardless of model output. |
| 18 | `btc_general/positive` | `btc_general/neutral` | Model error under the locked operational definition: newly accepting BTC directly increases payment access/adoption. |

The only recommended fixture correction is case 17 scope. It is recorded here rather than silently changing `benchmark_v2.json`; the v2 baseline therefore remains reproducible.
