#!/usr/bin/env python3
"""
Print current case + latest heartbeat from a Cursor terminal log file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Show current running test case from terminal log.")
    parser.add_argument(
        "--terminal-log",
        default="/Users/muskan/.cursor/projects/Users-muskan-BTP/terminals/2.txt",
        help="Path to terminal log file (default: terminal 2).",
    )
    args = parser.parse_args()

    log_path = Path(args.terminal_log)
    if not log_path.exists():
        print(f"Error: terminal log not found: {log_path}")
        return 1

    lines = log_path.read_text().splitlines()

    case = None
    for line in lines:
        m = re.search(r"\[\d+\]\s+n=\d+,\s+p=\d+,\s+seed=\d+", line)
        if m:
            case = m.group(0)

    last_running = next((line.strip() for line in reversed(lines) if "[running]" in line), "N/A")

    print(f"Current case: {case or 'N/A'}")
    print(f"Latest heartbeat: {last_running}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

