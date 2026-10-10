#!/usr/bin/env python3
"""Create and score the frozen v3 48-article validation set without DB writes."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import btc_sentiment as core
import score_v2 as impl
import score_v3


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_previous_urls(path):
    urls = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "result":
                urls.add(row["item"]["url"])
    return urls


def select(args):
    output = Path(args.output)
    if output.exists():
        raise SystemExit("selection output already exists")
    excluded = load_previous_urls(args.exclude)
    uri = "file:" + str(Path(args.db).resolve()) + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    query = """SELECT url, source, published, modified, collected, title, summary, text_hash
               FROM articles
               WHERE substr(published, 1, 10) BETWEEN '2025-01-01' AND '2025-12-31'"""
    rows = [dict(row) for row in connection.execute(query)]
    connection.close()
    chosen = []
    availability = {}
    for source in sorted({row["source"] for row in rows}):
        for month_number in range(1, 13):
            month = f"2025-{month_number:02d}"
            candidates = [row for row in rows if row["source"] == source
                          and row["published"][:7] == month and row["url"] not in excluded]
            candidates.sort(key=lambda row: hashlib.sha256(row["url"].encode()).hexdigest())
            availability[f"{source}/{month}"] = len(candidates)
            for row in candidates[:2]:
                row["id"] = hashlib.sha256(row["url"].encode()).hexdigest()
                row["selection_rank"] = candidates.index(row) + 1
                chosen.append(row)
    manifest = {
        "type": "selection_manifest",
        "method": "For each source and publication month, exclude every URL in sample_v3_results.jsonl, sort remaining URLs by ascending SHA-256 hex digest, and take the first two.",
        "database_opened_read_only": str(Path(args.db)),
        "excluded_manifest": str(Path(args.exclude)),
        "excluded_url_count": len(excluded),
        "requested_per_source_month": 2,
        "availability_after_exclusion": availability,
        "item_count": len(chosen),
        "items_hash": digest(chosen),
        "items": chosen,
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"selected {len(chosen)} articles; excluded {len(excluded)} prior URLs; wrote {output}")


def infer(args):
    output = Path(args.output)
    if output.exists():
        raise SystemExit("inference output already exists")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    references = json.loads(Path(args.references).read_text(encoding="utf-8"))
    items = manifest["items"]
    reference_by_id = {row["id"]: row for row in references["references"]}
    if set(reference_by_id) != {row["id"] for row in items}:
        raise SystemExit("reference IDs do not exactly match manifest IDs")
    prompt = score_v3.configure()
    model = next((model for model in core.local_api("/api/tags")["models"]
                  if model["name"] == args.model), None)
    if model is None:
        raise SystemExit("exact model name is not installed")
    show = core.local_api("/api/show", {"model": args.model})
    options = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 1024}
    metadata = {
        "version": impl.PROMPT_VERSION, "model": args.model, "model_digest": model["digest"],
        "prompt": prompt, "schema": impl.SCHEMA, "options": options,
        "think": False if "thinking" in show.get("capabilities", []) else None,
        "manifest_hash": digest(manifest), "references_hash": digest(references),
    }
    run_id = digest(metadata)
    valid = failed = 0
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "run", "run_id": run_id, "metadata": metadata}) + "\n")
        handle.flush()
        for item in items:
            text = impl.segments(item)
            request = {"model": args.model, "system": prompt,
                       "prompt": json.dumps({"segments": text}), "format": impl.SCHEMA,
                       "stream": False, "options": options}
            if metadata["think"] is not None:
                request["think"] = metadata["think"]
            raw = result = error = None
            try:
                raw = core.local_api("/api/generate", request)
                result = impl.validate(raw, text)
                valid += 1
            except Exception as exc:
                error = str(exc)
                failed += 1
            record = {"type": "result", "run_id": run_id, "item": item,
                      "reference": reference_by_id[item["id"]], "result": result,
                      "error": error, "raw_response": raw}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(item["id"], error or f'{result["scope"]}/{result["label"]}/{result["horizon"]}', flush=True)
    print(f"SUMMARY valid={valid} failed={failed} total={len(items)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    selection = subparsers.add_parser("select")
    selection.add_argument("--db", default="btc_sentiment.sqlite")
    selection.add_argument("--exclude", default="sample_v3_results.jsonl")
    selection.add_argument("--output", default="validation_v3_selection.json")
    selection.set_defaults(function=select)
    inference = subparsers.add_parser("infer")
    inference.add_argument("--manifest", default="validation_v3_selection.json")
    inference.add_argument("--references", default="validation_v3_references.json")
    inference.add_argument("--model", default="qwen3.5:9b")
    inference.add_argument("--output", default="validation_v3_results.jsonl")
    inference.set_defaults(function=infer)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
