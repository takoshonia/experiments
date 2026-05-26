#!/usr/bin/env python3
"""Evaluate Ollama postcorrection on a JSONL of (prediction, reference) pairs.

For each row:
  - Send `prediction` (STT1_Text) to Ollama with the postcorrection prompt
  - Read back the corrected text + token counts + duration
  - Compute WER/CER between `reference` (Text) and Ollama output

Outputs:
  - per-row JSONL (--output)
  - per-row XLSX (--xlsx)
  - summary JSON (--summary)

Usage (on AWS, inside the experiments venv):
  export OLLAMA_MODEL="gemma3:4b"
  python scripts/eval_postcorrection.py \
    --input manifest_sample.jsonl \
    --output results.jsonl \
    --xlsx results.xlsx \
    --summary summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

try:
    from openpyxl import Workbook
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: openpyxl. Install with: pip install openpyxl"
    ) from exc

from geostt_correct.ollama_backend import FEW_SHOT, SYSTEM_PROMPT, USER_TEMPLATE


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


def _call_ollama(host: str, model: str, prediction: str, temperature: float, timeout_s: int) -> dict:
    """POST one correction request, return raw fields incl. response + token counts."""
    user_prompt = USER_TEMPLATE.format(few_shot=FEW_SHOT, chunk=prediction.strip())
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = host.rstrip("/") + "/api/generate"
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json; charset=utf-8"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with stem/reference/prediction.")
    parser.add_argument("--output", type=Path, default=Path("results.jsonl"), help="Per-row results JSONL.")
    parser.add_argument("--xlsx", type=Path, default=Path("results.xlsx"), help="Per-row results XLSX.")
    parser.add_argument("--summary", type=Path, default=Path("summary.json"), help="Aggregate summary JSON.")
    parser.add_argument("--max-rows", type=int, default=None, help="Process at most N rows.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N rows (0 = silent).")
    parser.add_argument("--host", type=str, default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--model", type=str, default=os.environ.get("OLLAMA_MODEL", "gemma3:4b"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=int, default=300)
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
    print(f"Model: {args.model} @ {args.host}", flush=True)

    headers = [
        "stem",
        "reference",
        "prediction",
        "ollama_output",
        "wer_before",
        "wer_after",
        "wer_delta",
        "cer_before",
        "cer_after",
        "cer_delta",
        "prompt_tokens",
        "output_tokens",
        "total_tokens",
        "execution_time_s",
        "ollama_total_s",
        "error",
    ]
    wb = Workbook()
    ws = wb.active
    ws.title = "postcorrection"
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
            ollama_output = ""
            prompt_tokens = output_tokens = 0
            ollama_total_s = 0.0
            try:
                data = _call_ollama(
                    host=args.host,
                    model=args.model,
                    prediction=prediction,
                    temperature=args.temperature,
                    timeout_s=args.timeout_s,
                )
                ollama_output = str(data.get("response", "")).strip()
                prompt_tokens = int(data.get("prompt_eval_count") or 0)
                output_tokens = int(data.get("eval_count") or 0)
                ollama_total_s = float(data.get("total_duration") or 0) / 1e9
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"
            except urllib.error.URLError as e:
                err = f"URLError: {e}"
            except Exception as e:  # noqa: BLE001
                err = str(e)
            execution_time_s = time.perf_counter() - t0

            wer_after = _wer(reference, ollama_output) if not err else wer_before
            cer_after = _cer(reference, ollama_output) if not err else cer_before

            result = {
                "stem": stem,
                "reference": reference,
                "prediction": prediction,
                "ollama_output": ollama_output,
                "wer_before": wer_before,
                "wer_after": wer_after,
                "wer_delta": wer_after - wer_before,
                "cer_before": cer_before,
                "cer_after": cer_after,
                "cer_delta": cer_after - cer_before,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
                "execution_time_s": execution_time_s,
                "ollama_total_s": ollama_total_s,
                "error": err,
            }
            results.append(result)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            ws.append([result[h] for h in headers])

            if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(rows)):
                wall = time.perf_counter() - started
                tag = "ERR" if err else "OK"
                print(
                    f"[{i}/{len(rows)}] {tag} stem={stem} "
                    f"wer {wer_before:.2f}->{wer_after:.2f} "
                    f"cer {cer_before:.2f}->{cer_after:.2f} "
                    f"tok={prompt_tokens + output_tokens} "
                    f"t={execution_time_s:.2f}s wall={wall:.1f}s"
                    + (f" ERROR={err[:80]}" if err else ""),
                    flush=True,
                )

    wb.save(args.xlsx)

    ok = [r for r in results if not r["error"]]
    summary = {
        "input": str(args.input),
        "model": args.model,
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
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in ok),
        "total_output_tokens": sum(r["output_tokens"] for r in ok),
        "total_tokens": sum(r["total_tokens"] for r in ok),
        "average_execution_time_s": statistics.fmean(r["execution_time_s"] for r in ok) if ok else None,
        "total_execution_time_s": sum(r["execution_time_s"] for r in results),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n--- summary ---", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
