#!/usr/bin/env python3
"""Filter a results JSONL down to 'hard' rows for re-running postcorrection.

Reads a JSONL (e.g. results_500.jsonl from eval_postcorrection.py, or a
manifest sample with wer_before populated) and writes a new JSONL with only
rows that have STT errors worth correcting.

Output format matches what eval_postcorrection.py expects as input:
  {"stem", "reference", "prediction", "wer_before"?}

Usage (on AWS):
  python scripts/filter_hard_rows.py \
    --input results_500.jsonl \
    --output hard_rows.jsonl \
    --min-wer 0.30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL (results or manifest).")
    parser.add_argument("--output", type=Path, default=Path("hard_rows.jsonl"), help="Filtered output JSONL.")
    parser.add_argument("--min-wer", type=float, default=0.30, help="Keep rows with wer_before >= this (default 0.30).")
    parser.add_argument("--max-wer", type=float, default=None, help="Optional upper bound on wer_before.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows written (after filter).",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    kept: list[dict] = []
    total = 0
    skipped_no_wer = 0
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            wer = row.get("wer_before")
            if wer is None:
                skipped_no_wer += 1
                continue
            if wer < args.min_wer:
                continue
            if args.max_wer is not None and wer > args.max_wer:
                continue
            kept.append(
                {
                    "stem": row.get("stem", ""),
                    "reference": row.get("reference", ""),
                    "prediction": row.get("prediction", ""),
                    "wer_before": wer,
                }
            )

    if args.max_rows is not None:
        kept = kept[: args.max_rows]

    with args.output.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Input rows: {total}")
    print(f"Skipped (no wer_before): {skipped_no_wer}")
    rng_desc = f">= {args.min_wer}" + (f" and <= {args.max_wer}" if args.max_wer is not None else "")
    print(f"Kept (wer_before {rng_desc}): {len(kept)}")
    print(f"Wrote -> {args.output}")


if __name__ == "__main__":
    main()
