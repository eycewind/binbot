# Bollinger Event, Walk-State & CFAR-Inspired Detection Research

**Research contract · Version 0.1 · 14 October 2026**  
**Status:** Proposed methodology; no demonstrated trading edge  
**Primary instrument:** BTCUSDT, with later cross-asset validation  
**Primary input:** Validated UTC 5-minute OHLCV, candle times and completeness preserved

## 1. Research objective
Determine whether causal Bollinger-band events and multi-timescale market states predict *future, executable directional opportunities*, and whether additional context can reduce false triggers while retaining true opportunities. Optimize risk-adjusted net economic value subject to operational constraints, rather than maximizing in-sample classification accuracy.

**Non-goals:** visually pleasing labels; maximizing agreement between resolutions; treating rearming as a profitable reversal; guaranteed CFAR; adding RSI, volume or sentiment before price-only baselines have been validated.

## 2. What we have established vs. what remains unproven

### Established engineering findings
- Baseline `bollinger_bands()` creates complete, UTC-close-labelled bars. `evaluate_band_rejections()` evaluates against the *previous completed* band and has a reproducible candle-level baseline (1H: 897 support tests and 831 resistance tests in the previously examined annual run).
- E0/E0.1 contact cycles reduce repeated counting, but are not independent observations and `REARMED` is not a true-reversal outcome.
- E0.2/E03 use three labels: `NEUTRAL`, `LOWER_WALK`, `UPPER_WALK`. The currently inspected detector uses three qualifying bars to confirm and two non-near-band bars to exit, in units of candles rather than physical time.
- 2025 diagnostic completed-walk median durations (hours): 1D lower 96, upper 120; 1H lower 5, upper 4; 15m lower 1.25, upper 1; 5m both about 0.333. These are *confirmed-state durations*, not total development times, and exclude active/censored spells.
- A visually delayed 1H upper-walk confirmation on 2025-11-23 at 03:00 UTC followed a broken candidate run; the 15m detector can react earlier but produces more short-lived states.
- The November 2025 charts were debugging examples, not a representative evaluation sample. One earlier apparent timeout discrepancy was caused by a 15m-vs-1H dataframe mismatch; the two 1H configurations later matched exactly.

### Not yet established
- Walk-state detection accuracy, calibration, false-alarm probability, profit or stability across market regimes.
- Whether an `UPPER_WALK` or `LOWER_WALK` is predictive beyond momentum, Bollinger %B or volatility alone.
- Whether walk age, near-band occupancy, band slope, or changing band width predict the end of a walk or subsequent directional breakout.
- Whether 4H, 1H, 15m, 5m each contribute incremental signal, or are mostly redundant.

## 3. Timeframe and information architecture
Use **4H → 1H → 15m → 5m** as the principal timeframe ladder; 1D is optional macro context, not the primary adjacent timescale. Independently compute each timeframe's Bollinger features and states from its own *fully completed* candles, using 20 closing observations as the initial native definition. Align states on a 5-minute master timeline by carrying forward only already-completed states; no future coarse-bar completion may leak backward. Track `feature_available_at`, `candidate_time`, `confirm_time`, `exit_time` separately. Carry warm-up prior to the evaluation interval; preserve missing intervals and data-quality flags.

The 20-bar lookback covers different physical histories on different resolutions. This is intentional for native multiscale hypotheses, **not** an equivalent-filter comparison. Run an auxiliary fixed-physical-lookback sensitivity test when comparing detector invariance.

## 4. Event vocabulary and noncircular labels
Keep distinct:
1. **Contact:** an eligible candle crosses a previously known band, including wick-based excursions.
2. **Candidate:** price first meets a specified, contemporaneously observable proximity/slope/penetration condition.
3. **Confirmed walk:** predeclared persistence condition met at close; never backdate this as an available trading signal.
4. **Walk exit:** state ends due to a specified distance/persistence rule; exit is **not** automatically reversal.
5. **Market outcome:** objectively measured forward returns/barrier hits, independent of the detection definition.

For each candidate, retain rejected, never-confirmed, confirmed, unresolved and censored cases. A candidate judged by its later confirmation introduces selection bias. Record candidate de-duplication and cooldown; do not let repeated 5m ticks in the same price impulse masquerade as independent opportunities.

## 5. Milestone E0.3 — validate and characterize the walk detector (NEXT)

### E0.3a — Instrumentation and reproducibility
- Freeze detector revision/config, data interval, hash/fingerprint, relevant Git commit and exact environment.
- Add causal measurement columns to every completed candle: `pct_b`, band width, normalized upper/lower band slope, near-band flag, qualification flag, candidate-run length, current confirmed walk, state age, candidate start, confirmation/exit times and reasons.
- Test identical repeated runs; timestamp alignment; missing-data handling; warm-up; start/end truncation invariance; no-lookahead via prefix-versus-full-series comparison; boundary behavior at each resolution.
- Render full-history summary charts with date-range selectors; keep November as one fixed *diagnostic case*, not a tuning target.

### E0.3b — Walk reliability, timing and dependence
- Compute occupancy, candidate→confirmation fraction, candidate→timeout/abandonment fraction, confirmation lag and preconfirmation directional price displacement.
- Measure post-confirmation continuation/exit outcomes at predeclared horizons, with next-executable-price returns and market/transaction costs separately.
- Estimate walk survival and discrete exit hazard **conditional on surviving to age a**: `P(exit during next k bars | active at age a)`. Account for right-censoring; do not compare only completed walks.
- Measure same-direction repeated walks and cross-timeframe overlap/conditional transitions using states aligned causally. Distinguish time-weighted overlap from event-count overlap, and use episode/block bootstrap or regime-based folds for dependence.

### E0.3c — Bollinger compression / breakout study
- Define normalized band width: `(upper-lower)/middle`, and **past-only** rolling percentile of width, with a chosen minimum history and explicit exclusions around missing data.
- Separately define a squeeze candidate, squeeze duration, a *direction-free* breakout trigger, and the subsequent directional outcome. A narrow band forecasts neither bullish nor bearish direction by definition.
- Compare `P(breakout within H | low-width percentile)` against unconditional/base-rate probability; measure time to breakout, breakout direction, continuation, whipsaw, and executable net outcome.
- Test whether changes in width, not merely low absolute width, add information. Avoid selecting compression thresholds using the held-out period.

### E0.3d — Threshold sensitivity and adaptive detection
- Benchmark current fixed near-band, slope, consecutive-bar and exit thresholds against a **small predeclared** grid and a past-only adaptive percentile/volatility-scaled alternative.
- Evaluate candidate volume, recall/Pd, false-alarm rate among candidate opportunities, false alerts/day, median signal delay, duration calibration, and economic outcomes. Compare performance across volatility regimes.
- Do not declare adaptive thresholds better unless gains persist on untouched chronological folds with acceptable lost-opportunity cost.

**E0.3 acceptance gate:** causal replay tests pass; duration and censoring accounting verified; no silently excluded candidate outcomes; full-history and at least two out-of-sample market regimes assessed; fixed-versus-adaptive comparisons reported with uncertainty; remaining failure cases documented. Only then freeze the walk detector or explicitly retain uncertainty as a measured feature.

## 6. Milestone E04 — CFAR-inspired false-trigger study (AFTER E0.3)

**D0 — Unfiltered baseline:** Freeze deduplicated 15m candidates first (5m as separate sensitivity); compare event types (breakout, failed break, and walk candidate) rather than pooling them without labels.

**D1 — Context:** Join past-known 5m/15m/1H/4H states and state age. Compare all candidates, including countertrend signals; never impose rigid higher-timeframe agreement without evidence. Include simple price-momentum and volatility baselines before claiming a unique multiscale advantage.

**D2 — Filtering:** Use a simple interpretable score/filter first, with chronological development/validation/untouched test segmentation. Compare ROC and precision-recall, sensitivity (Pd), candidate-conditional Pfa, accepted signals/day, median delay, missed opportunities and net economic returns. Evaluate fixed, realistic Pfa constraints (e.g. 5%, 10%, 20%, 30%) as experiments, not promised guarantees.

**D3 — Adaptive filtering:** If fixed filters prove unstable, use only past-known local score distributions/reference windows. Quantile thresholds are *CFAR-inspired*, not guaranteed true CFAR under changing market dependence and regime shifts.

## 7. Ground-truth design — choose before modeling
Use event-relative first-passage barriers in ATR or realized-volatility units frozen at detection time, comparing upward/downward excursions from the **next executable price** over predeclared horizons (e.g., 3h, 6h, 12h, 24h). Report which barrier is hit first, neither hit, both within same OHLC interval (ordering ambiguous), unavailable execution price, or censored due to dataset end/gap. Separately track forward returns, maximum favorable/adverse excursions and net returns after costs. Labeling must not rely on subsequent detector state to decide whether the originating detection was “true.” Test both directional continuation and reversal hypotheses in clearly separate strata.

## 8. Evaluation hygiene
- Use calendar-chronological, embargoed/purged folds appropriate to maximum label horizon. Keep an untouched test segment; no repeated peeking for threshold selection.
- Estimate uncertainty by temporally blocked sampling or episode-level clustering. Walks and their constituent fine-timeframe signals are not IID.
- Report sample counts for each denominator, class prevalence, confidence intervals, activity/censoring, alert latency, and costs. Accuracy alone is insufficient.
- Verify features are known at signal decision time and orders occur no earlier than subsequent executable prices.
- Negative/null results are valid deliverables. No indicator family may be introduced without documenting the incremental hypothesis, baseline and tested outcome.

## 9. Later feature roadmap (not yet authorized as model inputs)
RSI raw value → RSI hysteretic region/transition/persistence → volume and aggressor flow where properly sourced → volatility state → sentiment with timestamped availability. Add one representation/family at a time and test incrementally against identical labels and splits. Simple lagged returns/volatility must remain comparison baselines.

## 10. Decisions, hypotheses and open questions

**Decisions**
- Prefer 4H/1H/15m/5m ladder over 1D/1H/15m/5m.
- High-resolution candidates may be screened by coarse context, but countertrend candidates must remain in the research set.
- E0.3 must precede E04. First establish detector fidelity and signal delay.
- Keep price-only measurements before introducing additional indicator families.

**Pre-registered hypotheses (not findings)**
- H1: Candidate continuation/reversal rates depend on confirmed walk age and near-band occupancy.
- H2: Small, historically unusual Bollinger bandwidth precedes a higher rate of future *volatility breakouts*, but not necessarily a known direction or profitable trade.
- H3: Coarse-timeframe state and state age improve the precision/Pd–Pfa trade-off of fine-timeframe candidates beyond price-only momentum/volatility baselines.
- H4: A past-only adaptive threshold improves cross-regime stability over one fixed threshold without sacrificing too much Pd or net expectancy.

**Open choices to freeze before E0.3 testing**
- Precise definitions of `near-band`, qualifying slope, candidate start, timeout and exit.
- Minimum warm-up for rolling band-width percentile.
- Candidate de-duplication/rearm/cooldown and true-outcome barrier units.
- Evaluation horizon, cost assumptions, which untouched date range, and what is considered a meaningful improvement.

## 11. Reproduction and checkpoint record
At each accepted milestone record: date, Git branch/commit, detector version, conda environment, data hash/coverage, parameter config, tests and their results, plots, tables, failures, interpretation and *next authorized change*. Commit tested changes at logical checkpoints. Keep this contract in Git; update with explicit decision amendments rather than silently rewriting the hypotheses after seeing results.

**Current checkpoint:** E03 plots reviewed; detector state persistence varies greatly across resolutions; predictive accuracy unknown. **Next authorized work: E0.3a instrumentation, causal tests, and reproducible full-history baseline.**