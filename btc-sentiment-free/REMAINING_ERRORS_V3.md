# Remaining v3 error analysis and pre-registered regressions

This review uses the frozen v3 outputs already present at commit `43510c37`. It does not use the fresh validation outputs.

## Error taxonomy

| Source | Case/row | Error class | Reference | v3 | Daily effect |
|---|---|---|---|---|---|
| Synthetic | 07 | Inclusion/exclusion | `unrelated/uncertain` | `relative_only/uncertain` | None here: both scopes are excluded. The distinction matters diagnostically but not to the daily denominator. |
| Synthetic | 10 | Scope subtype | `usd_explicit/neutral` | `btc_general/neutral` | None: both subtypes are included and neutral contributes 0 while remaining in the denominator. |
| Synthetic | 13 | Sentiment label | `btc_general/uncertain` | `btc_general/positive` | Yes: the unattributed combined flow is incorrectly included as +1 instead of excluded as uncertain. |
| Real sample | 1 | Inclusion/exclusion | `unrelated/uncertain` | `relative_only/uncertain` | None here because both are excluded; it exposes the same passing-analogy scope confusion as case 07. |
| Real sample | 5 | Scope subtype (ambiguous row) | likely `usd_explicit/mixed/forward` | `btc_general/mixed/forward` | No scope effect; both are included. The mixed reading was accepted as an improvement but the row remains ambiguous. |
| Real sample | 6 | Horizon | `current_or_past` | `forward` | None in the current daily formula. |
| Real sample | 8 | Horizon | `current_or_past` | `forward` | None in the current daily formula. |
| Real sample | 15 | Scope subtype | `usd_explicit/positive/forward` | `btc_general/positive/forward` | None: `$100K` should trigger `usd_explicit`, but both scopes aggregate identically. |
| Real sample | 23 | Horizon | `current_or_past` | `forward` | None in the current daily formula. |

The known remaining set contains no unambiguous positive-vs-negative, positive-vs-neutral, negative-vs-neutral, or mixed-vs-single-direction error on the reviewed real sample. The substantive label failure is synthetic case 13: `uncertain` versus `positive`. Ambiguous real row 5 concerns whether a concessive positive demand clause and bearish price clause require `mixed`; v3 did return `mixed`, so it is not counted as a remaining label error.

`usd_explicit` and `btc_general` are separate audit subtypes but are both included by the daily calculation. Horizon is also audit metadata and is not used by that calculation. Daily sentiment changes only when (a) an article crosses between an included scope (`usd_explicit`/`btc_general`) and an excluded scope, or (b) an included article changes among positive, negative, neutral, mixed, and uncertain. Positive/negative affect numerator and denominator; neutral/mixed affect only the denominator; uncertain and excluded scopes do not contribute.

## Targeted regression expectations

`benchmark_v3_targeted.json` records the expected outcomes before inference. Cases cover passing mentions, analogies, explicit USD steadiness, USD targets, unattributed combined flows, completed launches/rises/falls, and mixed concessive wording. Case 10 deliberately records two acceptable readings with the reason for ambiguity rather than forcing one answer.
