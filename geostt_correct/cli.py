from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geostt_correct.config import load_settings
from geostt_correct.pipeline import correct_document


def _read_text(path: str | None) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Georgian STT post-correction via local Ollama (constrained).")
    p.add_argument("input", nargs="?", default="-", help="Input file path, or - for stdin (default: -)")
    p.add_argument("-o", "--output", default="-", help="Output file, or - for stdout (default: -)")
    p.add_argument("--json-report", default=None, help="Write per-segment debug report JSON to this path")
    p.add_argument("--verbose", action="store_true", help="Print segment decisions to stderr")
    args = p.parse_args(argv)

    settings = load_settings()
    text = _read_text(args.input)
    result = correct_document(text, settings)

    out = result.text
    if args.output == "-":
        sys.stdout.write(out)
        if out and not out.endswith("\n"):
            sys.stdout.write("\n")
    else:
        Path(args.output).write_text(out, encoding="utf-8")

    if args.json_report:
        payload = {
            "ollama_model": settings.ollama_model,
            "segments": [
                {
                    "source": s.source,
                    "pre_llm": s.pre_llm,
                    "output": s.output,
                    "skipped_llm": s.skipped_llm,
                    "skip_reason": s.skip_reason,
                    "rejected_model": s.rejected_model,
                    "reject_reason": s.reject_reason,
                }
                for s in result.segments
            ],
        }
        Path(args.json_report).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.verbose:
        for i, s in enumerate(result.segments):
            tag = "SKIP" if s.skipped_llm else ("REJECT" if s.rejected_model else "OK")
            reason = s.skip_reason or s.reject_reason
            sys.stderr.write(f"[{i}] {tag} {reason}\n")
            sys.stderr.write(f"  IN : {s.source}\n")
            if s.pre_llm:
                sys.stderr.write(f"  PRE: {s.pre_llm}\n")
            sys.stderr.write(f"  OUT: {s.output}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
