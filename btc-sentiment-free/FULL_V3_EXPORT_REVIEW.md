# Frozen v3 completed-export review

This is a read-only audit of the completed frozen-v3 exports. No inference, collection, prompt/classifier change, training, or trading backtest was run.

## Reconciliation

The results database records run `7c2ad3400408cce53c1d0db4cc687350773b43058144d9163a8bbc1cf9701672`, model `qwen3.5:9b`, digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`, and frozen article-set hash `4e6afe42fbe3771ab1c8629ea01fa8fe8207914bbecc598ac891e7fff5ec0dc6`.

Independent reconstruction from `full_v3_runtime/articles.csv` found:

- 809 articles and 809 successful outcomes; zero failures or unreconciled inputs.
- 702 usable: 372 positive, 160 negative, 139 neutral, and 31 mixed.
- 92 excluded and 15 uncertain.
- Zero exact-normalized same-day headline duplicates in this frozen population.
- 365 unique, consecutive UTC dates from 2025-01-01 through 2025-12-31.
- Zero mismatches across every pooled and per-publisher count and score in `daily.csv`.

The score was recomputed as `100 * (positive - negative) / (positive + negative + neutral + mixed)`. There are 303 usable days and 62 blank-score days. Blank is distinct from a usable score of zero: 68 usable days have score exactly 0 (from neutral/mixed-only evidence or balanced positive and negative counts), whereas the 62 missing days have no usable denominator and remain empty.

## Daily coverage

| Month | Calendar days | Usable days | Missing days | Bitcoin.com usable days | CoinDesk usable days |
|---|---:|---:|---:|---:|---:|
| Jan | 31 | 22 | 9 | 18 | 14 |
| Feb | 28 | 23 | 5 | 23 | 12 |
| Mar | 31 | 24 | 7 | 18 | 14 |
| Apr | 30 | 23 | 7 | 22 | 11 |
| May | 31 | 28 | 3 | 28 | 15 |
| Jun | 30 | 26 | 4 | 22 | 14 |
| Jul | 31 | 26 | 5 | 23 | 15 |
| Aug | 31 | 25 | 6 | 22 | 14 |
| Sep | 30 | 27 | 3 | 23 | 14 |
| Oct | 31 | 29 | 2 | 27 | 10 |
| Nov | 30 | 23 | 7 | 22 | 13 |
| Dec | 31 | 27 | 4 | 24 | 14 |

The longest missing-score streak is four days, 2025-11-26 through 2025-11-29. Daily usable-article counts are: 0 on 62 days, 1 on 91, 2 on 103, 3 on 55, 4 on 32, 5 on 20, and 6 on 2. Among 303 usable days, 174 have one usable publisher and 129 have both.

Extreme pooled scores are common when evidence is sparse: +100 occurs on 90 days and -100 on 31. Of these, 43 positive extremes and 21 negative extremes are single-article days. Thus 64/121 extreme days—and 64/91 single-article days—are determined by one usable article.

## Publisher comparison

| Publisher | Inputs | Usable | Positive | Negative | Neutral | Mixed | Excluded | Uncertain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bitcoin.com | 559 | 496 | 276 (55.6%) | 96 (19.4%) | 103 (20.8%) | 21 (4.2%) | 55 | 8 |
| CoinDesk | 250 | 206 | 96 (46.6%) | 64 (31.1%) | 36 (17.5%) | 10 (4.9%) | 37 | 7 |

Both publishers have usable scores on 129 dates. Their daily score signs agree on 60 and disagree on 69; 31 disagreements are strictly positive versus negative, while 38 involve one publisher at zero. On those overlap dates, mean daily scores are 34.92 for Bitcoin.com and 15.31 for CoinDesk; medians are 50 and 0, and mean absolute difference is 73.62 points. These are descriptive coverage/model-output differences, not evidence by themselves that either publisher was classified incorrectly.

Bitcoin.com supplies 496/702 usable articles (70.7%) and CoinDesk 206/702 (29.3%). On overlap dates Bitcoin.com has more usable articles on 56 days, CoinDesk on 15, and counts are equal on 58. On all 58 dates where counts and publisher scores both differ, the pooled score is closer to the higher-volume publisher. The overlap-date pooled mean is 26.67 versus 25.12 under an equal-publisher daily average; the mean absolute daily shift from equal weighting is 8.72 points. This quantifies the intended article-weighted pooling effect without treating publisher imbalance as model error.

## Timestamp limitations

- Published and collected timestamps parse for all 809 articles. Modification timestamps are absent for 67; absence is not proof that no revision occurred.
- Of 742 supplied modification timestamps, 512 equal publication time, 211 occur after publication, and 19 precede publication (metadata anomalies that should not be interpreted as revisions).
- Among the 211 post-publication modifications: 175 occur on the same UTC date; 18 cross into the next UTC date while remaining under 24 hours after publication; 11 are 1–7 days later; 5 are 8–30 days later; 1 is 31–180 days later; and 1 is over 180 days later. The maximum delay is 309.07 days.
- 36 modifications occur at or after nominal D+1 00:00 UTC. The collected-text median lag from publication is 462.90 days (maximum 645.35), underscoring that this retrospective corpus is **not point-in-time safe**.

## Diagnostic plots and limitations

- `full_v3_daily_diagnostics.png`: separate panels for pooled daily sentiment (missing dates are gaps) and usable article counts by publisher.
- `full_v3_monthly_coverage.png`: pooled and per-publisher usable-day coverage by month.

The corpus includes only collected CoinDesk and Bitcoin.com items admitted by the source-selection pipeline; it is not a complete news census. Classification uses headlines and publisher summaries only. Deduplication removes exact normalized headlines within a UTC day, but does not collapse paraphrases, syndication, or multiple articles about the same event. Publisher volume is coverage, not independence.

## Proposed future intraday join

For a later, separate analysis, define day D sentiment as nominally available at **D+1 00:00 UTC**. Carry the score, positive/negative/neutral/mixed counts, excluded/uncertain/failed counts, usable count, publisher count, and missing-score indicator as separate fields. Do not replace a blank score with zero and do not forward-fill across missing days. Do not join these exports to trading data in this step. D+1 timing is a research assumption, not verified historical availability.

## Reproduce

```bash
conda run -n binbot python --version
MPLCONFIGDIR=/tmp/full_v3_mpl conda run -n binbot python analyze_full_v3_exports.py
```

The machine-readable audit is `FULL_V3_EXPORT_AUDIT.json`; large frozen runtime inputs remain unchanged under the ignored `full_v3_runtime/` directory.
