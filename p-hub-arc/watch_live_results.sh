#!/usr/bin/env bash
set -euo pipefail

# Periodically refresh live completed-results summary while run is active.
#
# Usage:
#   bash watch_live_results.sh
#   bash watch_live_results.sh /path/to/terminal.txt 30 live_completed_results.txt
#
# Args:
#   $1 terminal log path (default: terminal 2 file)
#   $2 refresh interval seconds (default: 30)
#   $3 output summary file (default: live_completed_results.txt)

TERMINAL_LOG="${1:-/Users/muskan/.cursor/projects/Users-muskan-BTP/terminals/2.txt}"
INTERVAL="${2:-30}"
OUT_FILE="${3:-live_completed_results.txt}"

while true; do
  date
  python3 extract_live_results.py --terminal-log "$TERMINAL_LOG" --out "$OUT_FILE"
  python3 monitor_current_case.py --terminal-log "$TERMINAL_LOG"
  echo "----"
  sleep "$INTERVAL"
done

