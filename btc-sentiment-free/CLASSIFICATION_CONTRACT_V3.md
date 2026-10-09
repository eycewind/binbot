# BTC/USD headline/summary classification contract — v3

This contract applies only to the supplied headline and publisher summary. It measures expressed BTC sentiment, not future returns, article-body sentiment, factual truth, or trading value.

## Scope precedence

1. `usd_explicit`: BTC has a USD price level (for example `$100K`) or an explicitly USD-denominated BTC price move/outlook. A dollar amount for an ETF flow, company purchase, liquidation, or financing is not a BTC price quote.
2. `btc_general`: BTC has an independent absolute price direction/outlook, demand, adoption/access, or regulation statement, but no explicit BTC/USD price as defined above. Absolute rises/falls remain here even if other assets are listed.
3. `relative_only`: the only BTC claim is an actual relative-performance, pair, dominance, or comparison claim. A passing mention or analogy is not enough.
4. `unrelated`: the subject is another token, company/stock, product, or person and no separate BTC claim exists.
5. `uncertain`: supplied text is too incomplete or ambiguous to establish target scope.

## Labels

- `positive`: explicit BTC price rise/bullish outlook or directly increased BTC demand, adoption, or access.
- `negative`: explicit BTC price fall/bearish outlook or directly decreased/rejected BTC demand, adoption, or access.
- `neutral`: an established BTC fact with no directional signal, including steady price. It is not a fallback for missing decisive information.
- `mixed`: both positive and negative BTC signals are supplied. Keep both signals even when they concern different horizons or one is less prominent.
- `uncertain`: the decisive claim is missing, truncated, ambiguous, or cannot be attributed to BTC. Excluded scopes always receive this label.

BTC-attributed ETF inflows/outflows are positive/negative `btc_general`. Combined-asset flows without a BTC breakdown are `btc_general/uncertain`. Evidence IDs guarantee exact source provenance only; the selected text must also logically support the scope, label, and (for `mixed`) both directions.

Horizons describe the explicit temporal claim: `current_or_past`, `forward`, `both`, or `unspecified`. A completed launch or purchase is current/past; it is not forward merely because the model infers future effects.

## Manual-review policy

Reviewers may mark a case ambiguous instead of forcing a gold label. Benchmark expectation corrections require a written semantic justification; model agreement is never sufficient justification.
