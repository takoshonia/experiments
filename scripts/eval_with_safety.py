#!/usr/bin/env python3
"""Evaluate the full postcorrection pipeline (gating + LLM + safety filter)
on a JSONL of (prediction, reference) pairs.

Unlike `eval_postcorrection.py`, this routes each row through
`geostt_correct.pipeline.correct_document`, so:
  - clean inputs may be skipped (gating)
  - risky LLM rewrites are rejected (safety filter -> keep original)

Outputs:
  - per-row JSONL (--output)
  - per-row XLSX (--xlsx)
  - summary JSON (--summary)

Usage (AWS):
  export OLLAMA_MODEL="gemma3:4b"
  python scripts/eval_with_safety.py \
    --input medium_rows.jsonl \
    --output results_safety.jsonl \
    --xlsx results_safety.xlsx \
    --summary summary_safety.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import unicodedata
from pathlib import Path

try:
    from openpyxl import Workbook
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: openpyxl. Install with: pip install openpyxl"
    ) from exc

from geostt_correct.config import load_settings
from geostt_correct.pipeline import correct_document


def _normalize_for_scoring(text: str) -> str:
    no_punct = "".join(
        ch for ch in text if not unicodedata.category(ch).startswith("P")
    )
    return " ".join(no_punct.split())


def _levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, y in enumerate(b, start=1):
            cost = 0 if x == y else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _wer(ref: str, hyp: str) -> float:
    r = _normalize_for_scoring(ref).split()
    h = _normalize_for_scoring(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(r, h) / len(r)


def _cer(ref: str, hyp: str) -> float:
    r = list(_normalize_for_scoring(ref))
    h = list(_normalize_for_scoring(hyp))
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(r, h) / len(r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results_safety.jsonl"))
    parser.add_argument("--xlsx", type=Path, default=Path("results_safety.xlsx"))
    parser.add_argument("--summary", type=Path, default=Path("summary_safety.json"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    rows: list[dict] = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    settings = load_settings()
    print(f"Loaded {len(rows)} rows", flush=True)
    print(f"Model: {settings.ollama_model} @ {settings.ollama_host}", flush=True)

    headers = [
        "stem",
        "reference",
        "prediction",
        "corrected",
        "wer_before",
        "wer_after",
        "wer_delta",
        "cer_before",
        "cer_after",
        "cer_delta",
        "any_llm_skipped",
        "skip_reasons",
        "any_model_rejected",
        "reject_reasons",
        "execution_time_s",
        "error",
    ]
    wb = Workbook()
    ws = wb.active
    ws.title = "postcorrection_safety"
    ws.append(headers)

    results: list[dict] = []
    started = time.perf_counter()

    with args.output.open("w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows, start=1):
            stem = row.get("stem", f"row_{i}")
            reference = row.get("reference", "")
            prediction = row.get("prediction", "")

            wer_before = _wer(reference, prediction)
            cer_before = _cer(reference, prediction)

            t0 = time.perf_counter()
            err = ""
            corrected = prediction
            skip_reasons: list[str] = []
            reject_reasons: list[str] = []
            any_skipped = False
            any_rejected = False
            try:
                doc = correct_document(prediction, settings)
                corrected = doc.text or prediction
                for seg in doc.segments:
                    if seg.skipped_llm:
                        any_skipped = True
                        if seg.skip_reason:
                            skip_reasons.append(seg.skip_reason)
                    if seg.rejected_model:
                        any_rejected = True
                        if seg.reject_reason:
                            reject_reasons.append(seg.reject_reason)
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
            execution_time_s = time.perf_counter() - t0

            wer_after = _wer(reference, corrected)
            cer_after = _cer(reference, corrected)

            result = {
                "stem": stem,
                "reference": reference,
                "prediction": prediction,
                "corrected": corrected,
                "wer_before": wer_before,
                "wer_after": wer_after,
                "wer_delta": wer_after - wer_before,
                "cer_before": cer_before,
                "cer_after": cer_after,
                "cer_delta": cer_after - cer_before,
                "any_llm_skipped": any_skipped,
                "skip_reasons": ",".join(sorted(set(skip_reasons))),
                "any_model_rejected": any_rejected,
                "reject_reasons": ",".join(sorted(set(reject_reasons))),
                "execution_time_s": execution_time_s,
                "error": err,
            }
            results.append(result)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            ws.append([result[h] for h in headers])

            if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(rows)):
                wall = time.perf_counter() - started
                tag = "ERR" if err else ("SKIP" if any_skipped else ("REJECT" if any_rejected else "OK"))
                print(
                    f"[{i}/{len(rows)}] {tag} stem={stem} "
                    f"wer {wer_before:.2f}->{wer_after:.2f} "
                    f"cer {cer_before:.2f}->{cer_after:.2f} "
                    f"t={execution_time_s:.2f}s wall={wall:.1f}s"
                    + (f" reasons={result['skip_reasons'] or result['reject_reasons']}" if (any_skipped or any_rejected) else "")
                    + (f" ERROR={err[:80]}" if err else ""),
                    flush=True,
                )

    wb.save(args.xlsx)

    ok = [r for r in results if not r["error"]]
    summary = {
        "input": str(args.input),
        "rows_processed": len(results),
        "rows_failed": len(results) - len(ok),
        "wer_before_mean": statistics.fmean(r["wer_before"] for r in ok) if ok else None,
        "wer_after_mean": statistics.fmean(r["wer_after"] for r in ok) if ok else None,
        "wer_before_median": statistics.median(r["wer_before"] for r in ok) if ok else None,
        "wer_after_median": statistics.median(r["wer_after"] for r in ok) if ok else None,
        "cer_before_mean": statistics.fmean(r["cer_before"] for r in ok) if ok else None,
        "cer_after_mean": statistics.fmean(r["cer_after"] for r in ok) if ok else None,
        "cer_before_median": statistics.median(r["cer_before"] for r in ok) if ok else None,
        "cer_after_median": statistics.median(r["cer_after"] for r in ok) if ok else None,
        "rows_wer_improved": sum(1 for r in ok if r["wer_after"] < r["wer_before"]),
        "rows_wer_worse": sum(1 for r in ok if r["wer_after"] > r["wer_before"]),
        "rows_wer_unchanged": sum(1 for r in ok if r["wer_after"] == r["wer_before"]),
        "rows_any_llm_skipped": sum(1 for r in ok if r["any_llm_skipped"]),
        "rows_any_model_rejected": sum(1 for r in ok if r["any_model_rejected"]),
        "average_execution_time_s": statistics.fmean(r["execution_time_s"] for r in ok) if ok else None,
        "total_execution_time_s": sum(r["execution_time_s"] for r in results),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n--- summary ---", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
