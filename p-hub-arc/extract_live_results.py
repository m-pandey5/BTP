#!/usr/bin/env python3
"""
Extract completed test results from a terminal log while the run is still active.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract completed timing results from terminal log.")
    parser.add_argument(
        "--terminal-log",
        default="/Users/muskan/.cursor/projects/Users-muskan-BTP/terminals/2.txt",
        help="Path to terminal log file (default: terminal 2).",
    )
    parser.add_argument(
        "--out",
        default="live_completed_results.txt",
        help="Output summary file path.",
    )
    args = parser.parse_args()

    src = Path(args.terminal_log)
    out = Path(args.out)

    if not src.exists():
        print(f"Error: terminal log not found: {src}")
        return 1

    lines = src.read_text().splitlines()
    rows = []
    case = None

    for i, line in enumerate(lines):
        m = re.search(r"\[(\d+)\]\s+n=(\d+),\s+p=(\d+),\s+seed=(\d+)", line)
        if m:
            case = f"[{m.group(1)}] n={m.group(2)}, p={m.group(3)}, seed={m.group(4)}"
        if "Obj:" in line:
            timing_line = lines[i - 1].strip() if i > 0 else ""
            rows.append((case or "[1] n=30, p=6, seed=42", timing_line, line.strip()))

    text = [f"Completed tests: {len(rows)}", ""]
    for c, t, o in rows:
        text += [c, f"  {t}", f"  {o}", ""]

    out.write_text("\n".join(text).rstrip() + "\n")
    print(f"Saved: {out.resolve()}")
    print(f"Completed tests found: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

