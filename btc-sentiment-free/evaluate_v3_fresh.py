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


def regress(args):
    """Run the frozen prompt on pre-registered synthetic cases."""
    output = Path(args.output)
    if output.exists():
        raise SystemExit("regression output already exists")
    items = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    prompt = score_v3.configure()
    model = next((model for model in core.local_api("/api/tags")["models"]
                  if model["name"] == args.model), None)
    if model is None:
        raise SystemExit("exact model name is not installed")
    show = core.local_api("/api/show", {"model": args.model})
    options = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 1024}
    metadata = {"version": impl.PROMPT_VERSION, "model": args.model,
                "model_digest": model["digest"], "prompt": prompt, "schema": impl.SCHEMA,
                "options": options, "think": False if "thinking" in show.get("capabilities", []) else None,
                "cases_hash": digest(items)}
    run_id = digest(metadata)
    valid = failed = accepted = 0
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "run", "run_id": run_id, "metadata": metadata}) + "\n")
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
            match = bool(result and any(all(result[key] == value for key, value in expected.items())
                                        for expected in item["expected"]))
            accepted += int(match)
            handle.write(json.dumps({"type": "result", "run_id": run_id, "item": item,
                                     "result": result, "error": error,
                                     "matches_accepted_expected": match,
                                     "raw_response": raw}, ensure_ascii=False) + "\n")
            handle.flush()
            print(item["id"], error or f'{result["scope"]}/{result["label"]}/{result["horizon"]}', match)
    print(f"SUMMARY valid={valid} failed={failed} accepted={accepted}/{len(items)}")


def report(args):
    output = Path(args.output)
    if output.exists():
        raise SystemExit("report output already exists")
    with open(args.results, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    run = next(row for row in rows if row["type"] == "run")
    results = [row for row in rows if row["type"] == "result"]
    included = {"usd_explicit", "btc_general"}
    valid = sum(row["result"] is not None and row["error"] is None for row in results)
    inclusion_agree = label_agree = scope_agree = horizon_agree = 0
    reference_included = reference_excluded = ambiguous = 0
    disagreements = []
    for number, row in enumerate(results, 1):
        result = row["result"]
        reference = row["reference"]
        expected = reference["expected"]
        ambiguous += int(reference["ambiguous"])
        expected_included = all(choice["scope"] in included for choice in expected)
        reference_included += int(expected_included)
        reference_excluded += int(not expected_included)
        model_included = bool(result and result["scope"] in included)
        inclusion_ok = result is not None and expected_included == model_included
        label_ok = bool(result and any(result["label"] == choice["label"] for choice in expected))
        scope_ok = bool(result and any(result["scope"] == choice["scope"] for choice in expected))
        horizon_ok = bool(result and any(result["horizon"] == choice["horizon"] for choice in expected))
        inclusion_agree += int(inclusion_ok)
        label_agree += int(expected_included and label_ok)
        scope_agree += int(scope_ok)
        horizon_agree += int(horizon_ok)
        dimensions = []
        if not inclusion_ok:
            dimensions.append("inclusion/exclusion")
        if expected_included and not label_ok:
            dimensions.append("sentiment label")
        if not scope_ok:
            dimensions.append("scope subtype")
        if not horizon_ok:
            dimensions.append("horizon")
        if dimensions:
            disagreements.append((number, row, dimensions))
    model_excluded = sum(not row["result"] or row["result"]["scope"] not in included for row in results)
    lines = [
        "# Frozen v3 fresh-sample validation report", "",
        "The selection and reference labels were committed in `d2157aba` before this inference run. The references are agent-reviewed labels, not independently established human ground truth. No prompt or label was changed after viewing these outputs.", "",
        "## Run and denominators", "",
        f"- Run ID: `{run['run_id']}`",
        f"- Model/digest: `{run['metadata']['model']}` / `{run['metadata']['model_digest']}`",
        f"- Structural validity: **{valid}/{len(results)}**. Failed responses: **{len(results)-valid}**; all {len(results)} selected records remain in structural, inclusion, scope, and horizon denominators.",
        f"- Inclusion/exclusion agreement: **{inclusion_agree}/{len(results)}**.",
        f"- Sentiment-label agreement on reference-included articles: **{label_agree}/{reference_included}**. A failed or excluded model response to a reference-included article would count as disagreement.",
        f"- Exact scope-subtype agreement: **{scope_agree}/{len(results)}**.",
        f"- Horizon agreement: **{horizon_agree}/{len(results)}**.",
        f"- Ambiguous reference items: **{ambiguous}**. Reference-excluded items: **{reference_excluded}**; model-excluded items: **{model_excluded}**.",
        "", "Agreement for an ambiguous item means agreement with at least one pre-registered accepted outcome. Scope-subtype agreement is stricter than inclusion agreement: `usd_explicit` and `btc_general` both enter the daily calculation. Horizon does not enter it. Positive/negative contribute ±1 and enter the denominator; neutral/mixed contribute 0 and enter it; uncertain or excluded items do not enter it.", "",
        "## Every substantive disagreement", "",
        f"There are **{len(disagreements)}** items with at least one assessed dimension disagreement. Full supplied text is reproduced below; no article-body or external information was used.", ""
    ]
    for number, row, dimensions in disagreements:
        item, reference, result = row["item"], row["reference"], row["result"]
        expected_text = "; ".join(f"`{x['scope']}/{x['label']}/{x['horizon']}`" for x in reference["expected"])
        model_text = "failed: `" + str(row["error"]) + "`" if result is None else f"`{result['scope']}/{result['label']}/{result['horizon']}`"
        lines.extend([
            f"### {number}. {item['title']}", "",
            f"- Disagreed dimensions: {', '.join(dimensions)}",
            f"- Reference: {expected_text}",
            f"- Frozen v3: {model_text}",
            f"- Headline: {item['title']}",
            f"- Summary: {item.get('summary') or '[empty]'}",
            f"- Reference reasoning: {reference['reason']}",
            f"- v3 reasoning: {result['reason'] if result else row['error']}", ""
        ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}: {len(disagreements)} disagreements")


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
    regression = subparsers.add_parser("regress")
    regression.add_argument("--cases", default="benchmark_v3_targeted.json")
    regression.add_argument("--model", default="qwen3.5:9b")
    regression.add_argument("--output", default="benchmark_v3_targeted_results.jsonl")
    regression.set_defaults(function=regress)
    reporting = subparsers.add_parser("report")
    reporting.add_argument("--results", default="validation_v3_results.jsonl")
    reporting.add_argument("--output", default="VALIDATION_V3_REPORT.md")
    reporting.set_defaults(function=report)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
