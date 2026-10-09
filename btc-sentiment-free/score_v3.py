#!/usr/bin/env python3
"""Prompt-v3 entry point for the local BTC/USD classifier pilot."""
import score_v2 as impl


def configure():
    prompt = impl.PROMPT
    prompt = prompt.replace(
        "usd_explicit: a BTC dollar price, USD move or USD outlook is explicitly given (including headline).\n"
        "btc_general: statements about BTC itself, its demand, adoption, regulation or outlook without a BTC USD quote.\n"
        "relative_only: ONLY relative BTC performance/dominance against crypto, gold or other assets; no independent BTC USD or absolute outlook.",
        "usd_explicit: a BTC dollar PRICE LEVEL or explicitly USD-denominated BTC PRICE move/outlook is given (including headline).\n"
        "btc_general: statements about BTC itself, its demand, adoption, regulation or absolute price direction/outlook without a BTC USD price level or explicit USD price denomination.\n"
        "relative_only: ONLY an actual comparison of BTC performance/dominance against crypto, gold or other assets; no independent BTC USD or absolute outlook. Merely mentioning BTC among assets is not a relative comparison.")
    prompt = prompt.replace(
        "ETF flows qualify only if BTC flows are identified; combined BTC/ETH totals without attribution are insufficient.\n"
        "Mention of ETH does not exclude a separate BTC statement. Stable BTC while altcoins fall is neutral, not bullish.",
        "Dollar-denominated ETF flows are fund flows, not a BTC dollar-price quote, so they use btc_general scope. ETF flows qualify directionally only if BTC flows are identified; combined BTC/ETH totals without attribution require label uncertain, not neutral.\n"
        "Mention of ETH does not exclude a separate BTC statement. Stable BTC while altcoins fall is neutral, not bullish.\n"
        "An absolute BTC rise, fall, rally, decline, bullish cue or bearish cue remains in scope even without a dollar quote and even when other assets are also listed.\n"
        "Explicitly increased BTC payment access/adoption is positive; explicitly reduced or rejected access/adoption is negative. Do not require a price forecast for those labels.\n"
        "If the text contains both a positive BTC demand/adoption signal and a bearish BTC price/outlook signal (or the reverse), label mixed and cite evidence for both.\n"
        "When a sentence is incomplete before its decisive predicate, use uncertain scope and uncertain label unless other supplied text independently establishes scope and label.")
    if prompt == impl.PROMPT:
        raise RuntimeError('v3 prompt patch did not apply')
    impl.PROMPT = prompt
    impl.PROMPT_VERSION = 'pilot_v3'
    return prompt


if __name__ == '__main__':
    configure()
    impl.main()
