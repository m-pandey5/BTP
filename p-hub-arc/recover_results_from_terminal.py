#!/usr/bin/env python3
"""
Recover completed timing-test results from a terminal log file.

Works even if the run was interrupted (e.g., KeyboardInterrupt), as long as
the completed-case summary lines exist in terminal output.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional


CASE_RE = re.compile(r"\[(\d+)\]\s+n=(\d+),\s+p=(\d+),\s+seed=(\d+)")
TIMING_RE = re.compile(
    r"F3=(?P<F3>[0-9.]+|N/A)s?,\s*"
    r"NormB=(?P<NormB>[0-9.]+|N/A)s,\s*"
    r"NewB=(?P<NewB>[0-9.]+|N/A)s"
    r"(?:,\s*MD=(?P<MD>[0-9.]+|N/A)s,\s*MDP=(?P<MDP>[0-9.]+|N/A)s,\s*P12=(?P<P12>[0-9.]+|N/A)s)?\s*"
    r"\(fastest:\s*(?P<fastest>[A-Za-z0-9_-]+)\)"
)
OBJ_RE = re.compile(
    r"Obj:\s*F3=(?P<F3>[0-9.]+|N/A)\s+"
    r"NormB=(?P<NormB>[0-9.]+|N/A)\s+"
    r"NewB=(?P<NewB>[0-9.]+|N/A)"
    r"(?:\s+MD=(?P<MD>[0-9.]+|N/A)\s+MDP=(?P<MDP>[0-9.]+|N/A)\s+P12=(?P<P12>[0-9.]+|N/A))?\s+\|\s+(?P<match>Match|MISMATCH)"
)


def _to_float_or_none(v: Optional[str]) -> Optional[float]:
    if v is None or v == "N/A":
        return None
    return float(v)


def recover(log_path: Path) -> List[Dict]:
    lines = log_path.read_text(errors="replace").splitlines()
    results: List[Dict] = []
    current_case: Optional[Dict] = None
    pending_timing: Optional[Dict] = None

    for line in lines:
        case_m = CASE_RE.search(line)
        if case_m:
            current_case = {
                "idx": int(case_m.group(1)),
                "n": int(case_m.group(2)),
                "p": int(case_m.group(3)),
                "seed": int(case_m.group(4)),
            }
            pending_timing = None
            continue

        t_m = TIMING_RE.search(line)
        if t_m and current_case is not None:
            pending_timing = {
                "f3_time": _to_float_or_none(t_m.group("F3")),
                "norm_time": _to_float_or_none(t_m.group("NormB")),
                "new_time": _to_float_or_none(t_m.group("NewB")),
                "md_time": _to_float_or_none(t_m.group("MD")),
                "mdp_time": _to_float_or_none(t_m.group("MDP")),
                "p12_time": _to_float_or_none(t_m.group("P12")),
                "fastest": t_m.group("fastest"),
            }
            continue

        o_m = OBJ_RE.search(line)
        if o_m and current_case is not None:
            row = dict(current_case)
            if pending_timing:
                row.update(pending_timing)
            row.update(
                {
                    "f3_obj": _to_float_or_none(o_m.group("F3")),
                    "norm_obj": _to_float_or_none(o_m.group("NormB")),
                    "new_obj": _to_float_or_none(o_m.group("NewB")),
                    "md_obj": _to_float_or_none(o_m.group("MD")),
                    "mdp_obj": _to_float_or_none(o_m.group("MDP")),
                    "p12_obj": _to_float_or_none(o_m.group("P12")),
                    "match": o_m.group("match"),
                }
            )
            results.append(row)
            pending_timing = None

    # Deduplicate by case id if repeated in pasted logs; keep latest.
    by_key: Dict[tuple, Dict] = {}
    for r in results:
        key = (r["idx"], r["n"], r["p"], r["seed"])
        by_key[key] = r
    ordered = sorted(by_key.values(), key=lambda x: x["idx"])
    return ordered


def write_outputs(rows: List[Dict], out_prefix: Path) -> None:
    txt_path = out_prefix.with_suffix(".txt")
    csv_path = out_prefix.with_suffix(".csv")

    txt_lines = [f"Recovered completed tests: {len(rows)}", ""]
    for r in rows:
        txt_lines.append(f"[{r['idx']}] n={r['n']}, p={r['p']}, seed={r['seed']}")
        txt_lines.append(
            "  "
            f"F3={r.get('f3_time')}s, NormB={r.get('norm_time')}s, NewB={r.get('new_time')}s, "
            f"MD={r.get('md_time')}s, MDP={r.get('mdp_time')}s, P12={r.get('p12_time')}s "
            f"(fastest: {r.get('fastest', 'N/A')})"
        )
        txt_lines.append(
            "  "
            f"Obj: F3={r.get('f3_obj')} NormB={r.get('norm_obj')} NewB={r.get('new_obj')} "
            f"MD={r.get('md_obj')} MDP={r.get('mdp_obj')} P12={r.get('p12_obj')} | {r.get('match')}"
        )
        txt_lines.append("")
    txt_path.write_text("\n".join(txt_lines).rstrip() + "\n")

    fieldnames = [
        "idx",
        "n",
        "p",
        "seed",
        "f3_time",
        "norm_time",
        "new_time",
        "md_time",
        "mdp_time",
        "p12_time",
        "fastest",
        "f3_obj",
        "norm_obj",
        "new_obj",
        "md_obj",
        "mdp_obj",
        "p12_obj",
        "match",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"Saved text summary: {txt_path}")
    print(f"Saved csv summary:  {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover completed results from terminal log.")
    parser.add_argument("--terminal-log", required=True, help="Path to terminal output log text file.")
    parser.add_argument(
        "--out-prefix",
        default="recovered_large_run_results",
        help="Output file prefix (writes .txt and .csv).",
    )
    args = parser.parse_args()

    log_path = Path(args.terminal_log)
    if not log_path.exists():
        print(f"Error: terminal log not found: {log_path}")
        return 1

    rows = recover(log_path)
    write_outputs(rows, Path(args.out_prefix))
    print(f"Recovered rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

