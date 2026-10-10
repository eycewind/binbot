#!/usr/bin/env python3
"""Freeze, resumably score, and export the frozen v3 full-dataset baseline."""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3

import btc_sentiment as core
import score_v2 as impl
import score_v3

MODEL = "qwen3.5:9b"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 1024}
INCLUDED = {"usd_explicit", "btc_general"}
LABELS = ("positive", "negative", "neutral", "mixed")
UTC = dt.timezone.utc


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256((value if isinstance(value, str) else canonical(value)).encode()).hexdigest()


def configure():
    if impl.PROMPT_VERSION != "pilot_v3":
        score_v3.configure()
    return impl.PROMPT


def text_hash(item):
    return digest(item["title"] + "\n" + (item.get("summary") or ""))


def read_manifest(path):
    lines = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not lines or lines[0].get("type") != "manifest":
        raise ValueError("invalid manifest header")
    items = lines[1:]
    if any(row.get("type") != "article" for row in items):
        raise ValueError("invalid manifest article")
    calculated = digest(items)
    if calculated != lines[0]["articles_hash"] or len(items) != lines[0]["article_count"]:
        raise ValueError("manifest count/hash mismatch")
    if len({row["url"] for row in items}) != len(items):
        raise ValueError("duplicate manifest URL")
    for row in items:
        if row["input_text_hash"] != text_hash(row):
            raise ValueError("manifest input text hash mismatch: " + row["url"])
    return lines[0], items


def freeze(args):
    backup = Path(args.backup)
    manifest = Path(args.manifest)
    summary = Path(args.summary)
    for path in (backup, manifest, summary):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
    backup.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{Path(args.db).resolve()}?mode=ro", uri=True)
    target = sqlite3.connect(backup)
    source.backup(target)
    target.commit(); target.close(); source.close()
    frozen = sqlite3.connect(f"file:{backup.resolve()}?mode=ro", uri=True)
    frozen.row_factory = sqlite3.Row
    rows = [dict(row) for row in frozen.execute(
        "SELECT url,source,published,modified,collected,title,summary,text_hash FROM articles ORDER BY url")]
    frozen.close()
    items = []
    coverage = {}
    for row in rows:
        item = {"type": "article", **row, "input_text_hash": text_hash(row)}
        if item["text_hash"] != item["input_text_hash"]:
            raise SystemExit("stored text hash mismatch: " + item["url"])
        items.append(item)
        key = f'{item["source"]}/{item["published"][:7]}'
        coverage[key] = coverage.get(key, 0) + 1
    header = {"type": "manifest", "format": 1, "article_count": len(items),
              "articles_hash": digest(items), "order": "URL ascending",
              "source_database": str(Path(args.db)), "frozen_backup": str(backup)}
    with manifest.open("x", encoding="utf-8") as handle:
        handle.write(canonical(header) + "\n")
        for item in items:
            handle.write(canonical(item) + "\n")
    info = {"expected_article_count": 809, "article_count": len(items),
            "count_matches_expected": len(items) == 809, "articles_hash": header["articles_hash"],
            "manifest_sha256": digest(manifest.read_text(encoding="utf-8")),
            "backup_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(), "coverage": coverage}
    summary.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(info, sort_keys=True))


def live_identity():
    prompt = configure()
    tags = core.local_api("/api/tags").get("models", [])
    model = next((row for row in tags if row.get("name") == MODEL), None)
    if model is None:
        raise RuntimeError(f"required local model not installed: {MODEL}")
    show = core.local_api("/api/show", {"model": MODEL})
    think = False if "thinking" in show.get("capabilities", []) else None
    return {"version": impl.PROMPT_VERSION, "model": MODEL, "model_digest": model["digest"],
            "prompt": prompt, "schema": impl.SCHEMA, "options": OPTIONS, "think": think}


def identity_from_jsonl(path):
    prompt = configure()
    run = metadata_from_jsonl(path)
    meta = run["metadata"]
    expected = {"version": impl.PROMPT_VERSION, "model": MODEL, "prompt": prompt,
                "schema": impl.SCHEMA, "options": OPTIONS}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(f"{path}: incompatible frozen {key}")
    if not meta.get("model_digest"):
        raise RuntimeError(f"{path}: missing model digest")
    return {**expected, "model_digest": meta["model_digest"], "think": meta.get("think")}


def expected_identity(manifest_header, live):
    value = {**live, "articles_hash": manifest_header["articles_hash"],
             "article_count": manifest_header["article_count"], "max_attempts_per_article": 1}
    return {**value, "run_id": digest(value)}


def open_results(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript("""
      CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS inputs(url TEXT PRIMARY KEY,input_text_hash TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY,at TEXT NOT NULL,url TEXT NOT NULL,
        input_text_hash TEXT NOT NULL,request TEXT NOT NULL,raw_response TEXT,error TEXT,provenance TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS outcomes(url TEXT PRIMARY KEY,input_text_hash TEXT NOT NULL,status TEXT NOT NULL,
        result TEXT,error TEXT,attempt_id INTEGER,provenance TEXT NOT NULL);
    """)
    return connection


def assert_state(connection, identity, items):
    stored = {row["key"]: json.loads(row["value"]) for row in connection.execute("SELECT * FROM metadata")}
    if not stored:
        for key, value in identity.items():
            connection.execute("INSERT INTO metadata VALUES(?,?)", (key, canonical(value)))
        connection.executemany("INSERT INTO inputs VALUES(?,?)",
                               [(row["url"], row["input_text_hash"]) for row in items])
        connection.commit()
        return
    if stored != identity:
        raise RuntimeError("result database identity differs; refusing resume")
    actual = [(row["url"], row["input_text_hash"]) for row in connection.execute("SELECT * FROM inputs ORDER BY url")]
    expected = [(row["url"], row["input_text_hash"]) for row in items]
    if actual != expected:
        raise RuntimeError("result database inputs differ; refusing resume")


def init(args):
    header, items = read_manifest(args.manifest)
    live = identity_from_jsonl(args.identity_jsonl) if args.identity_jsonl else live_identity()
    identity = expected_identity(header, live)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    connection = open_results(args.results)
    assert_state(connection, identity, items)
    connection.close()
    print(canonical(identity))


def metadata_from_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        first = json.loads(next(handle))
    if first.get("type") != "run":
        raise ValueError("missing JSONL run metadata")
    return first


def import_existing(args):
    header, items = read_manifest(args.manifest)
    by_url = {row["url"]: row for row in items}
    live = identity_from_jsonl(args.inputs[0])
    identity = expected_identity(header, live)
    connection = open_results(args.results)
    assert_state(connection, identity, items)
    imported = skipped = 0
    for filename in args.inputs:
        run = metadata_from_jsonl(filename)
        meta = run["metadata"]
        for key in ("version", "model", "model_digest", "prompt", "schema", "options", "think"):
            if meta.get(key) != live[key]:
                raise RuntimeError(f"{filename}: incompatible {key}")
        with open(filename, encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                record = json.loads(line)
                item = record["item"]
                frozen = by_url.get(item.get("url"))
                if frozen is None:
                    skipped += 1; continue
                if any(item.get(key) != frozen[key] for key in ("title", "summary")):
                    raise RuntimeError(f"{filename}: input differs for {item['url']}")
                if connection.execute("SELECT 1 FROM outcomes WHERE url=?", (item["url"],)).fetchone():
                    skipped += 1; continue
                raw = record.get("raw_response")
                text = impl.segments(frozen)
                result = impl.validate(raw, text) if record.get("result") is not None else None
                error = record.get("error")
                if result != record.get("result"):
                    raise RuntimeError(f"{filename}: stored result fails exact replay")
                provenance = canonical({"kind": "verified_existing_v3", "file": filename,
                                        "source_run_id": run["run_id"]})
                request = canonical({"reused": True, "segments": text})
                cursor = connection.execute("""INSERT INTO attempts(at,url,input_text_hash,request,raw_response,error,provenance)
                    VALUES(?,?,?,?,?,?,?)""", (core.now(), item["url"], frozen["input_text_hash"], request,
                    canonical(raw) if raw is not None else None, error, provenance))
                status = "success" if result is not None else "failed"
                connection.execute("INSERT INTO outcomes VALUES(?,?,?,?,?,?,?)",
                    (item["url"], frozen["input_text_hash"], status,
                     canonical(result) if result is not None else None, error, cursor.lastrowid, provenance))
                imported += 1
        connection.commit()
    connection.close()
    print(f"imported={imported} skipped={skipped}")


def score(args):
    header, items = read_manifest(args.manifest)
    live = live_identity()
    identity = expected_identity(header, live)
    connection = open_results(args.results)
    assert_state(connection, identity, items)
    completed = {row["url"] for row in connection.execute("SELECT url FROM outcomes")}
    pending = [row for row in items if row["url"] not in completed]
    if args.limit is not None:
        pending = pending[:args.limit]
    success = failed = 0
    for index, item in enumerate(pending, 1):
        text = impl.segments(item)
        request = {"model": MODEL, "system": live["prompt"],
                   "prompt": json.dumps({"segments": text}), "format": impl.SCHEMA,
                   "stream": False, "options": OPTIONS}
        if live["think"] is not None:
            request["think"] = live["think"]
        raw = result = error = None
        try:
            raw = core.local_api("/api/generate", request)
            result = impl.validate(raw, text)
        except Exception as exc:
            error = str(exc)
        provenance = canonical({"kind": "full_v3_inference", "run_id": identity["run_id"]})
        cursor = connection.execute("""INSERT INTO attempts(at,url,input_text_hash,request,raw_response,error,provenance)
            VALUES(?,?,?,?,?,?,?)""", (core.now(), item["url"], item["input_text_hash"], canonical(request),
            canonical(raw) if raw is not None else None, error, provenance))
        status = "success" if result is not None else "failed"
        connection.execute("INSERT INTO outcomes VALUES(?,?,?,?,?,?,?)",
            (item["url"], item["input_text_hash"], status,
             canonical(result) if result is not None else None, error, cursor.lastrowid, provenance))
        connection.commit()
        success += status == "success"; failed += status == "failed"
        print(f"{index}/{len(pending)} {status} {item['url']}", flush=True)
    total = connection.execute("SELECT count(*) FROM outcomes").fetchone()[0]
    connection.close()
    print(f"batch_success={success} batch_failed={failed} reconciled={total}/{len(items)}")


def parse_time(value):
    if not value:
        return None
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return parsed.astimezone(UTC)


def contribution(result):
    if result is None:
        return "failed_unscored", 0, 0
    if result["scope"] == "uncertain":
        return "uncertain", 0, 0
    if result["scope"] not in INCLUDED:
        return "excluded", 0, 0
    label = result["label"]
    return label, 1 if label == "positive" else -1 if label == "negative" else 0, int(label in LABELS)


def export(args):
    header, items = read_manifest(args.manifest)
    connection = open_results(args.results)
    metadata = {row["key"]: json.loads(row["value"]) for row in connection.execute("SELECT * FROM metadata")}
    if metadata.get("articles_hash") != header["articles_hash"]:
        raise RuntimeError("results/manifest identity mismatch")
    outcomes = {row["url"]: dict(row) for row in connection.execute("SELECT * FROM outcomes")}
    connection.close()
    if len(outcomes) != len(items):
        raise RuntimeError(f"refusing final export: only {len(outcomes)}/{len(items)} inputs have outcomes")
    article_path, daily_path, summary_path = map(Path, (args.articles, args.daily, args.summary))
    for path in (article_path, daily_path, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    article_rows = []
    timestamp_issues = []
    for item in items:
        outcome = outcomes.get(item["url"])
        result = json.loads(outcome["result"]) if outcome and outcome["result"] else None
        bucket, numerator, denominator = contribution(result)
        try:
            published = parse_time(item["published"])
            if published is None:
                timestamp_issues.append({"url": item["url"], "field": "published", "error": "missing"})
        except Exception as exc:
            published = None; timestamp_issues.append({"url": item["url"], "field": "published", "error": str(exc)})
        for field in ("modified", "collected"):
            try:
                parsed = parse_time(item[field])
                if parsed is None:
                    timestamp_issues.append({"url": item["url"], "field": field, "error": "missing"})
            except Exception as exc:
                timestamp_issues.append({"url": item["url"], "field": field, "error": str(exc)})
        article_rows.append({**item, "status": outcome["status"] if outcome else "unscored",
            "scope": result["scope"] if result else "", "label": result["label"] if result else "",
            "horizon": result["horizon"] if result else "", "error": outcome["error"] if outcome else "",
            "provenance": outcome["provenance"] if outcome else "", "daily_bucket": bucket,
            "daily_numerator": numerator, "daily_denominator": denominator,
            "published_date_utc": published.date().isoformat() if published else "",
            "nominal_next_day_utc": (published.date() + dt.timedelta(days=1)).isoformat() if published else "",
            "availability_note": "Research assumption: available on next UTC date; historical availability not verified."})
    article_fields = list(article_rows[0])
    sources = sorted({row["source"] for row in article_rows})
    daily_fields = ["date_utc", "nominal_next_day_utc", "availability_note", *LABELS,
                    "excluded", "uncertain", "failed_unscored", "usable_articles", "publisher_count",
                    "articles", "unique_headlines", "sentiment_score"]
    for source in sources:
        daily_fields += [f"{source}_{label}" for label in LABELS] + [f"{source}_excluded", f"{source}_uncertain",
            f"{source}_failed_unscored", f"{source}_usable_articles", f"{source}_sentiment_score"]
    by_day = {}
    for row in article_rows:
        by_day.setdefault(row["published_date_utc"], []).append(row)
    daily_rows = []
    for offset in range(365):
        day = dt.date(2025, 1, 1) + dt.timedelta(days=offset)
        rows = sorted(by_day.get(day.isoformat(), []), key=lambda row: (row["published"], row["url"]))
        seen = set(); unique = []
        for row in rows:
            key = core.norm(row["title"]).casefold()
            if key not in seen:
                seen.add(key); unique.append(row)
            else:
                row["daily_bucket"] = "duplicate_headline"
                row["daily_numerator"] = row["daily_denominator"] = 0
        values = {"date_utc": day.isoformat(), "nominal_next_day_utc": (day + dt.timedelta(days=1)).isoformat(),
                  "availability_note": "Research assumption only; historical availability not verified.",
                  "articles": len(rows), "unique_headlines": len(unique)}
        groups = [(None, unique)] + [(source, [row for row in unique if row["source"] == source]) for source in sources]
        for source, group in groups:
            prefix = "" if source is None else source + "_"
            counts = {name: sum(row["daily_bucket"] == name for row in group)
                      for name in (*LABELS, "excluded", "uncertain", "failed_unscored")}
            usable = sum(counts[label] for label in LABELS)
            score_value = "" if not usable else round(100 * (counts["positive"] - counts["negative"]) / usable, 4)
            for name, value in counts.items(): values[prefix + name] = value
            values[prefix + "usable_articles"] = usable
            values[prefix + "sentiment_score"] = score_value
        values["publisher_count"] = len({row["source"] for row in unique if row["daily_denominator"]})
        reconciled = sum(values[name] for name in (*LABELS, "excluded", "uncertain", "failed_unscored"))
        if reconciled != values["unique_headlines"]:
            raise RuntimeError(f"daily reconciliation failed for {day}: {reconciled} != {values['unique_headlines']}")
        daily_rows.append(values)
    with article_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=article_fields); writer.writeheader(); writer.writerows(article_rows)
    with daily_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=daily_fields); writer.writeheader(); writer.writerows(daily_rows)
    distribution = {name: sum(row["daily_bucket"] == name for row in article_rows) for name in
                    (*LABELS, "excluded", "uncertain", "failed_unscored", "duplicate_headline")}
    if sum(distribution.values()) != len(article_rows):
        raise RuntimeError("article reconciliation failed")
    publisher = {source: {name: sum(row["source"] == source and row["daily_bucket"] == name for row in article_rows)
                          for name in (*LABELS, "excluded", "uncertain", "failed_unscored", "duplicate_headline")}
                 for source in sources}
    modified_present = sum(bool(row["modified"]) for row in article_rows)
    revisions = sum(bool(row["modified"]) and row["modified"] != row["published"] for row in article_rows)
    summary = {"run_id": metadata.get("run_id"), "article_count": len(items),
               "outcome_count": len(outcomes), "unreconciled": len(items) - len(outcomes),
               "distribution_after_headline_dedup": distribution, "publisher_distribution": publisher,
               "publisher_input_articles": {source: sum(row["source"] == source for row in article_rows) for source in sources},
               "publisher_usable_articles": {source: sum(row["source"] == source and row["daily_denominator"] == 1 for row in article_rows) for source in sources},
               "days_with_usable": sum(row["usable_articles"] > 0 for row in daily_rows),
               "days_with_one_usable_article": sum(row["usable_articles"] == 1 for row in daily_rows),
               "days_with_no_usable_articles": sum(row["usable_articles"] == 0 for row in daily_rows),
               "missing_or_invalid_timestamps": timestamp_issues,
               "modified_timestamp_present": modified_present, "revisions_modified_differs_from_published": revisions,
               "point_in_time_warning": "Retrospectively collected current text is not point-in-time safe.",
               "availability_assumption": "Nominal next-day UTC availability is a research assumption, not verified historical availability."}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze"); p.add_argument("--db", default="btc_sentiment.sqlite"); p.add_argument("--backup", default="full_v3_runtime/frozen.sqlite"); p.add_argument("--manifest", default="FULL_V3_INPUT_MANIFEST.jsonl"); p.add_argument("--summary", default="FULL_V3_FREEZE.json"); p.set_defaults(function=freeze)
    p = sub.add_parser("init"); p.add_argument("--manifest", default="FULL_V3_INPUT_MANIFEST.jsonl"); p.add_argument("--results", default="full_v3_runtime/results.sqlite"); p.add_argument("--identity-jsonl", help="offline exact v3 run metadata source; scoring still re-verifies the live model"); p.set_defaults(function=init)
    p = sub.add_parser("import-existing"); p.add_argument("--manifest", default="FULL_V3_INPUT_MANIFEST.jsonl"); p.add_argument("--results", default="full_v3_runtime/results.sqlite"); p.add_argument("inputs", nargs="+"); p.set_defaults(function=import_existing)
    p = sub.add_parser("score"); p.add_argument("--manifest", default="FULL_V3_INPUT_MANIFEST.jsonl"); p.add_argument("--results", default="full_v3_runtime/results.sqlite"); p.add_argument("--limit", type=int); p.set_defaults(function=score)
    p = sub.add_parser("export"); p.add_argument("--manifest", default="FULL_V3_INPUT_MANIFEST.jsonl"); p.add_argument("--results", default="full_v3_runtime/results.sqlite"); p.add_argument("--articles", default="full_v3_runtime/articles.csv"); p.add_argument("--daily", default="full_v3_runtime/daily.csv"); p.add_argument("--summary", default="FULL_V3_BASELINE_SUMMARY.json"); p.set_defaults(function=export)
    args = parser.parse_args(); args.function(args)


if __name__ == "__main__":
    main()
