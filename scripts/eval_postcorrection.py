#!/usr/bin/env python3
"""Evaluate geostt-correct on a JSONL of (prediction, reference) pairs.

For each row: run prediction through correct_document (LLM postcorrection),
compute WER/CER before vs after, and write per-row + summary outputs.

Usage (AWS):
  python scripts/eval_postcorrection.py \
    --input manifest_sample.jsonl \
    --output results.jsonl \
    --summary summary.json

JSONL input rows must have: stem, reference, prediction.
Optional: wer_before, cer_before, duration_s.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import unicodedata
from pathlib import Path

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
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with stem/reference/prediction rows.")
    parser.add_argument("--output", type=Path, default=Path("results.jsonl"), help="Per-row results JSONL.")
    parser.add_argument("--summary", type=Path, default=Path("summary.json"), help="Aggregate summary JSON.")
    parser.add_argument("--max-rows", type=int, default=None, help="Process at most N rows (debug).")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N rows.")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    rows: list[dict] = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    print(f"Loaded {len(rows)} rows from {args.input}", flush=True)

    settings = load_settings()
    print(f"Model: {settings.ollama_model} @ {settings.ollama_host}", flush=True)

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
            try:
                doc = correct_document(prediction, settings)
                corrected = doc.text
                llm_applied = any(not s.skipped_llm for s in doc.segments)
                model_rejected = any(s.rejected_model for s in doc.segments)
            except Exception as exc:  # noqa: BLE001
                corrected = prediction
                llm_applied = False
                model_rejected = False
                err = str(exc)
            elapsed = time.perf_counter() - t0

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
                "llm_applied": llm_applied,
                "model_rejected": model_rejected,
                "elapsed_s": elapsed,
                "error": err,
            }
            results.append(result)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(rows)):
                wall = time.perf_counter() - started
                tag = "OK"
                if err:
                    tag = "ERR"
                elif not llm_applied:
                    tag = "SKIP"
                elif model_rejected:
                    tag = "REJECT"
                print(
                    f"[{i}/{len(rows)}] {tag} stem={stem} "
                    f"wer {wer_before:.2f}->{wer_after:.2f} "
                    f"cer {cer_before:.2f}->{cer_after:.2f} "
                    f"t={elapsed:.2f}s wall={wall:.1f}s"
                    + (f" ERROR={err[:80]}" if err else ""),
                    flush=True,
                )

    ok = [r for r in results if not r["error"]]
    summary = {
        "input": str(args.input),
        "ollama_model": settings.ollama_model,
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
        "rows_improved_wer": sum(1 for r in ok if r["wer_after"] < r["wer_before"]),
        "rows_worse_wer": sum(1 for r in ok if r["wer_after"] > r["wer_before"]),
        "rows_unchanged_wer": sum(1 for r in ok if r["wer_after"] == r["wer_before"]),
        "rows_llm_applied": sum(1 for r in ok if r["llm_applied"]),
        "rows_model_rejected": sum(1 for r in ok if r["model_rejected"]),
        "total_elapsed_s": sum(r["elapsed_s"] for r in results),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n--- summary ---", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
