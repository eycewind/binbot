#!/usr/bin/env python3
"""Recompute and audit completed frozen-v3 exports without inference or writes to inputs."""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics

LABELS = ("positive", "negative", "neutral", "mixed")
OTHER = ("excluded", "uncertain", "failed_unscored")
EXPECTED_RUN = "7c2ad3400408cce53c1d0db4cc687350773b43058144d9163a8bbc1cf9701672"


def norm(value):
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def file_sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_time(value):
    if not value:
        return None
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone missing")
    return parsed.astimezone(dt.timezone.utc)


def score(counts):
    usable = sum(counts[label] for label in LABELS)
    return None if not usable else round(100 * (counts["positive"] - counts["negative"]) / usable, 4)


def sign(value):
    return 0 if value == 0 else 1 if value > 0 else -1


def longest_streak(dates):
    best = current = []
    for date in dates:
        if current and date != current[-1] + dt.timedelta(days=1):
            current = []
        current = current + [date]
        if len(current) > len(best):
            best = current
    return {"days": len(best), "start": best[0].isoformat() if best else None,
            "end": best[-1].isoformat() if best else None}


def reconstruct(articles, sources):
    by_day = collections.defaultdict(list)
    for row in articles:
        by_day[row["published_date_utc"]].append(row)
    result = []
    for offset in range(365):
        day = dt.date(2025, 1, 1) + dt.timedelta(days=offset)
        rows = sorted(by_day.get(day.isoformat(), []), key=lambda row: (row["published"], row["url"]))
        seen = set(); unique = []
        for row in rows:
            key = norm(row["title"]).casefold()
            if key not in seen:
                seen.add(key); unique.append(row)
        output = {"date_utc": day.isoformat(), "articles": len(rows), "unique_headlines": len(unique)}
        for source, group in [(None, unique), *[(s, [r for r in unique if r["source"] == s]) for s in sources]]:
            prefix = "" if source is None else source + "_"
            counts = {bucket: sum(row["daily_bucket"] == bucket for row in group) for bucket in (*LABELS, *OTHER)}
            for bucket, count in counts.items(): output[prefix + bucket] = count
            output[prefix + "usable_articles"] = sum(counts[label] for label in LABELS)
            output[prefix + "sentiment_score"] = score(counts)
        output["publisher_count"] = sum(output[f"{source}_usable_articles"] > 0 for source in sources)
        result.append(output)
    return result


def compare_daily(actual, rebuilt, sources):
    fields = [*LABELS, *OTHER, "usable_articles", "publisher_count", "articles", "unique_headlines", "sentiment_score"]
    for source in sources:
        fields += [f"{source}_{name}" for name in (*LABELS, *OTHER, "usable_articles", "sentiment_score")]
    mismatches = []
    for a, b in zip(actual, rebuilt):
        if a["date_utc"] != b["date_utc"]:
            mismatches.append({"date": a["date_utc"], "field": "date_utc", "actual": a["date_utc"], "rebuilt": b["date_utc"]})
        for field in fields:
            raw = a[field]
            parsed = None if field.endswith("sentiment_score") and raw == "" else float(raw) if field.endswith("sentiment_score") else int(raw)
            if parsed != b[field]:
                mismatches.append({"date": a["date_utc"], "field": field, "actual": parsed, "rebuilt": b[field]})
    return mismatches


def timestamp_audit(articles):
    missing = collections.Counter(); invalid = []
    same_day = after_pub = after_nominal = before_pub = equal_time = 0
    deltas = []
    collected_deltas = []
    for row in articles:
        parsed = {}
        for field in ("published", "modified", "collected"):
            try:
                parsed[field] = parse_time(row[field])
                if parsed[field] is None: missing[field] += 1
            except Exception as exc:
                parsed[field] = None; invalid.append({"url": row["url"], "field": field, "error": str(exc)})
        published, modified, collected = parsed["published"], parsed["modified"], parsed["collected"]
        if published and modified and modified < published: before_pub += 1
        if published and modified and modified == published: equal_time += 1
        if published and modified and modified > published:
            after_pub += 1
            delta = (modified - published).total_seconds() / 86400
            deltas.append(delta)
            if modified.date() == published.date(): same_day += 1
            nominal = dt.datetime.combine(published.date() + dt.timedelta(days=1), dt.time(), dt.timezone.utc)
            if modified >= nominal: after_nominal += 1
        if published and collected:
            collected_deltas.append((collected - published).total_seconds() / 86400)
    return {"missing": dict(missing), "invalid": invalid, "modified_after_publication": after_pub,
            "modified_before_publication": before_pub, "modified_equal_to_publication": equal_time,
            "modified_same_utc_day_after_publication": same_day,
            "modified_after_nominal_next_day_utc": after_nominal,
            "modified_next_utc_day_but_under_24_hours_after": sum(0 < x < 1 for x in deltas) - same_day,
            "modified_1_to_7_days_after": sum(1 <= x <= 7 for x in deltas),
            "modified_8_to_30_days_after": sum(7 < x <= 30 for x in deltas),
            "modified_31_to_180_days_after": sum(30 < x <= 180 for x in deltas),
            "modified_over_180_days_after": sum(x > 180 for x in deltas),
            "max_modification_delay_days": round(max(deltas), 3) if deltas else None,
            "median_modification_delay_days": round(statistics.median(deltas), 3) if deltas else None,
            "collection_delay_days_median": round(statistics.median(collected_deltas), 3),
            "collection_delay_days_max": round(max(collected_deltas), 3)}


def plot(daily, monthly, sources, output_daily, output_monthly):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dates = [dt.date.fromisoformat(row["date_utc"]) for row in daily]
    scores = [math.nan if row["sentiment_score"] is None else row["sentiment_score"] for row in daily]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    axes[0].plot(dates, scores, color="#315b7d", linewidth=1.1)
    axes[0].axhline(0, color="black", linewidth=.6); axes[0].set_ylabel("Pooled sentiment")
    axes[0].set_title("Frozen v3 daily sentiment (missing days remain gaps)")
    axes[0].set_ylim(-105, 105); axes[0].grid(alpha=.2)
    bottom = [0] * len(dates)
    colors = ["#d9822b", "#4c78a8"]
    for source, color in zip(sources, colors):
        values = [row[f"{source}_usable_articles"] for row in daily]
        axes[1].bar(dates, values, bottom=bottom, width=1.0, label=source, color=color)
        bottom = [a+b for a,b in zip(bottom, values)]
    axes[1].set_ylabel("Usable articles"); axes[1].set_xlabel("UTC publication date")
    axes[1].set_title("Daily usable article counts by publisher"); axes[1].legend(); axes[1].grid(axis="y", alpha=.2)
    fig.savefig(output_daily, dpi=160); plt.close(fig)
    months = list(monthly)
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    x = list(range(len(months))); width=.25
    ax.bar([v-width for v in x], [monthly[m]["pooled_usable_days"] for m in months], width, label="pooled")
    for index, source in enumerate(sources):
        ax.bar([v + index*width for v in x], [monthly[m][source+"_usable_days"] for m in months], width, label=source)
    ax.set_xticks(x, [m[5:] for m in months]); ax.set_xlabel("2025 month"); ax.set_ylabel("Usable days")
    ax.set_title("Monthly usable-day coverage"); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.savefig(output_monthly, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", default="full_v3_runtime/articles.csv")
    parser.add_argument("--daily", default="full_v3_runtime/daily.csv")
    parser.add_argument("--results", default="full_v3_runtime/results.sqlite")
    parser.add_argument("--json", default="FULL_V3_EXPORT_AUDIT.json")
    parser.add_argument("--daily-plot", default="full_v3_daily_diagnostics.png")
    parser.add_argument("--monthly-plot", default="full_v3_monthly_coverage.png")
    args = parser.parse_args()
    with open(args.articles, newline="", encoding="utf-8") as handle: articles = list(csv.DictReader(handle))
    with open(args.daily, newline="", encoding="utf-8") as handle: actual = list(csv.DictReader(handle))
    sources = sorted({row["source"] for row in articles})
    connection = sqlite3.connect(f"file:{Path(args.results).resolve()}?mode=ro", uri=True); connection.row_factory=sqlite3.Row
    metadata = {row["key"]: json.loads(row["value"]) for row in connection.execute("SELECT * FROM metadata")}
    outcomes = {row["status"]: row["n"] for row in connection.execute("SELECT status,count(*) n FROM outcomes GROUP BY status")}
    connection.close()
    rebuilt = reconstruct(articles, sources)
    mismatches = compare_daily(actual, rebuilt, sources)
    dates = [dt.date.fromisoformat(row["date_utc"]) for row in rebuilt]
    consecutive = dates == [dt.date(2025,1,1)+dt.timedelta(days=i) for i in range(365)]
    distribution = collections.Counter(row["daily_bucket"] for row in articles)
    expected = {"positive":372,"negative":160,"neutral":139,"mixed":31,"excluded":92,"uncertain":15,"failed_unscored":0}
    usable_dates = [row for row in rebuilt if row["usable_articles"]]
    missing_dates = [dt.date.fromisoformat(row["date_utc"]) for row in rebuilt if not row["usable_articles"]]
    usable_count_distribution = dict(sorted(collections.Counter(row["usable_articles"] for row in rebuilt).items()))
    monthly = {}
    for month in range(1,13):
        key=f"2025-{month:02d}"; rows=[row for row in rebuilt if row["date_utc"].startswith(key)]
        monthly[key]={"calendar_days":len(rows),"pooled_usable_days":sum(row["usable_articles"]>0 for row in rows),
                      "missing_score_days":sum(row["usable_articles"]==0 for row in rows)}
        for source in sources: monthly[key][source+"_usable_days"]=sum(row[f"{source}_usable_articles"]>0 for row in rows)
    overlap = [row for row in rebuilt if all(row[f"{source}_usable_articles"] for source in sources)]
    sign_disagree = [row for row in overlap if sign(row[f"{sources[0]}_sentiment_score"]) != sign(row[f"{sources[1]}_sentiment_score"])]
    exact_opposite = [row for row in overlap if sign(row[f"{sources[0]}_sentiment_score"])*sign(row[f"{sources[1]}_sentiment_score"]) == -1]
    sign_pairs = collections.Counter(f'{sign(row[f"{sources[0]}_sentiment_score"]):+d}/{sign(row[f"{sources[1]}_sentiment_score"]):+d}' for row in overlap)
    count_relation = collections.Counter("equal" if row[f"{sources[0]}_usable_articles"]==row[f"{sources[1]}_usable_articles"] else
                                         sources[0]+"_more" if row[f"{sources[0]}_usable_articles"]>row[f"{sources[1]}_usable_articles"] else sources[1]+"_more" for row in overlap)
    equal_weight = [(row[f"{sources[0]}_sentiment_score"]+row[f"{sources[1]}_sentiment_score"])/2 for row in overlap]
    pooled = [row["sentiment_score"] for row in overlap]
    unequal = [row for row in overlap if row[f"{sources[0]}_usable_articles"] != row[f"{sources[1]}_usable_articles"] and row[f"{sources[0]}_sentiment_score"] != row[f"{sources[1]}_sentiment_score"]]
    closer_to_high_volume = 0
    for row in unequal:
        high = sources[0] if row[f"{sources[0]}_usable_articles"] > row[f"{sources[1]}_usable_articles"] else sources[1]
        low = sources[1] if high == sources[0] else sources[0]
        closer_to_high_volume += abs(row["sentiment_score"]-row[f"{high}_sentiment_score"]) < abs(row["sentiment_score"]-row[f"{low}_sentiment_score"])
    publisher = {}
    for source in sources:
        counts={label:sum(row["source"]==source and row["daily_bucket"]==label for row in articles) for label in (*LABELS,*OTHER)}
        scores=[row[f"{source}_sentiment_score"] for row in overlap]
        publisher[source]={"input_articles":sum(row["source"]==source for row in articles),"labels":counts,
                           "usable_articles":sum(counts[x] for x in LABELS),"overlap_daily_score_mean":round(statistics.mean(scores),4),
                           "overlap_daily_score_median":round(statistics.median(scores),4)}
    extremes={"positive_100":sum(row["sentiment_score"]==100 for row in usable_dates),
              "negative_100":sum(row["sentiment_score"]==-100 for row in usable_dates),
              "positive_100_single_article":sum(row["sentiment_score"]==100 and row["usable_articles"]==1 for row in usable_dates),
              "negative_100_single_article":sum(row["sentiment_score"]==-100 and row["usable_articles"]==1 for row in usable_dates)}
    audit={"run_identity":{"actual":metadata.get("run_id"),"expected":EXPECTED_RUN,"matches":metadata.get("run_id")==EXPECTED_RUN,
                           "model":metadata.get("model"),"model_digest":metadata.get("model_digest"),"articles_hash":metadata.get("articles_hash")},
           "audited_input_sha256":{"articles_csv":file_sha256(args.articles),"daily_csv":file_sha256(args.daily),"results_sqlite":file_sha256(args.results)},
           "reconciliation":{"articles":len(articles),"outcomes":outcomes,"daily_rows":len(rebuilt),"unique_dates":len(set(dates)),
                             "consecutive_2025":consecutive,"daily_field_mismatches":mismatches,
                             "article_distribution":dict(distribution),"expected_distribution":expected,
                             "expected_distribution_matches":all(distribution[k]==v for k,v in expected.items()),
                             "usable_articles":sum(distribution[k] for k in LABELS)},
           "daily":{"usable_days":len(usable_dates),"missing_score_days":len(missing_dates),
                    "usable_zero_score_days":sum(row["sentiment_score"] == 0 for row in usable_dates),
                    "one_usable_article_days":sum(row["usable_articles"]==1 for row in rebuilt),
                    "usable_count_distribution":usable_count_distribution,"longest_missing_streak":longest_streak(missing_dates),
                    "one_usable_publisher_days":sum(row["publisher_count"]==1 for row in usable_dates),
                    "two_usable_publisher_days":sum(row["publisher_count"]==2 for row in usable_dates),"extreme_scores":extremes,
                    "monthly":monthly},
           "publishers":{"by_source":publisher,"overlap_usable_dates":len(overlap),"sign_disagreement_dates":len(sign_disagree),
                         "strict_opposite_sign_dates":len(exact_opposite),"sign_agreement_dates":len(overlap)-len(sign_disagree),
                         "sign_pair_counts_first_bitcoincom_second_coindesk":dict(sign_pairs),
                         "count_relation_on_overlap":dict(count_relation),
                         "unequal_counts_and_scores_dates":len(unequal),"pooled_closer_to_higher_volume_publisher_dates":closer_to_high_volume,
                         "pooled_minus_equal_publisher_mean_abs":round(statistics.mean(abs(a-b) for a,b in zip(pooled,equal_weight)),4),
                         "publisher_score_mean_absolute_difference":round(statistics.mean(abs(row[f"{sources[0]}_sentiment_score"]-row[f"{sources[1]}_sentiment_score"]) for row in overlap),4),
                         "pooled_daily_mean_on_overlap":round(statistics.mean(pooled),4),
                         "equal_publisher_daily_mean_on_overlap":round(statistics.mean(equal_weight),4)},
           "timestamps":timestamp_audit(articles),
           "limitations":["Source selection contains only CoinDesk and Bitcoin.com coverage captured by the collector; it is not a complete news census.",
                          "Deduplication collapses exact normalized same-day headlines only; paraphrases and shared underlying events remain.",
                          "Only collected headline and publisher summary text is classified.",
                          "Retrospectively collected text is not point-in-time safe; absent modification metadata does not prove no revision.",
                          "Nominal D+1 00:00 UTC availability is an unverified research assumption."]}
    Path(args.json).write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    plot(rebuilt,monthly,sources,args.daily_plot,args.monthly_plot)
    print(json.dumps(audit,indent=2,sort_keys=True))


if __name__ == "__main__": main()
