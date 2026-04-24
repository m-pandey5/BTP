"""
Large-Instance Batch Timing: F3, Norm, New, plus optional MD and Pareto variants.

**Solvers (same random W, D per row; sequential, not parallel; each has its own
Gurobi TimeLimit, e.g. 600s per call — total wall time is the sum of all calls):**

  1) F3 (direct MIP) — skippable with LARGE_BENCH_SKIP_F3=1
  2) Normal Benders (HubArcBenders)
  3) New Benders (solve_benders_hub_arc, Phase1+2)
  4) McDaniel-Devine (md_benders_hub_arc) — unless LARGE_BENCH_NO_EXTRA=1
  5) MD + Pareto (md_benders_hub_arc_pareto, two_step)
  6) New-model Pareto phase12 (new_model_hub_arc_pareto_phase12, two_step)

**Quick mode (F3 + Norm + New only, no MD / MD Pareto / P12):**

  LARGE_BENCH_NO_EXTRA=1 python3 test_timing_large.py

**Skip F3 only** (e.g. large n, slow Python build):

  LARGE_BENCH_SKIP_F3=1 python3 test_timing_large.py

`match` in CSV: when extras run, all reported objectives (F3 if present, else
the five Benders) must agree within 1e-4.

Output: [large_instance_results.csv](large_instance_results.csv) in this folder
(one row per n,p,seed; all solver columns; failed cells empty/None).

Prerequisites: numpy, gurobipy. Run from p-hub-arc:  python3 test_timing_large.py
"""

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

# Fail fast with a clear message (import chain needs numpy)
try:
    import numpy  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    print(
        "Error: numpy is required. Activate your venv and run:\n"
        "  python3 -m pip install numpy",
        file=sys.stderr,
    )
    sys.exit(1)

from test_timing_comprehensive import (
    aggregate_by_instance,
    print_detailed_results,
    print_timing_table,
    run_timing_suite,
)


def get_large_instance_specs() -> List[Dict]:
    """Instance configs: one small n=30 smoke case first, then n=60..200 (two p each)."""
    configs = []
    # Smoke: runs first so you can confirm the script solves before the heavy n=60+ block
    configs.append({"n": 30, "p": 6, "fixed": False})
    for n, p1, p2 in [
        (60,  12, 15),
        (75,  15, 19),
        (100, 20, 25),
        (125, 25, 31),
        (150, 30, 38),
        (200, 40, 50),
    ]:
        configs.append({"n": n, "p": p1, "fixed": False})
        configs.append({"n": n, "p": p2, "fixed": False})
    return configs


def save_results_csv(results: List[Dict], path: str) -> None:
    """Save per-run results to a CSV (all solver time/obj/status columns + match)."""
    fieldnames = [
        "n", "p", "seed",
        "f3_time", "f3_obj", "f3_status",
        "norm_time", "norm_obj", "norm_status",
        "new_time", "new_obj", "new_status",
        "md_time", "md_obj", "md_status",
        "mdp_time", "mdp_obj", "mdp_status",
        "p12_time", "p12_obj", "p12_status",
        "match",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"\nResults saved to: {path}")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "y")


def _fastest_winner(
    r: Dict, include_extras: bool, skip_f3: bool
) -> Optional[str]:
    if "error" in r:
        return None
    times: List[Tuple[str, float]] = []
    if not skip_f3 and r.get("f3_time") is not None and r["f3_time"] >= 0:
        times.append(("F3", r["f3_time"]))
    for name, key in [
        ("NormB", "norm_time"),
        ("NewB", "new_time"),
    ]:
        t = r.get(key)
        if t is not None and t >= 0:
            times.append((name, t))
    if include_extras:
        for name, key in [
            ("MD", "md_time"),
            ("MDP", "mdp_time"),
            ("P12", "p12_time"),
        ]:
            t = r.get(key)
            if t is not None and t >= 0:
                times.append((name, t))
    if not times:
        return None
    return min(times, key=lambda x: x[1])[0]


def main():
    skip_f3 = _env_truthy("LARGE_BENCH_SKIP_F3")
    # Default: run MD + MD Pareto + P12 after New. Set LARGE_BENCH_NO_EXTRA=1 to skip.
    include_extras = not _env_truthy("LARGE_BENCH_NO_EXTRA")

    print("\n" + "=" * 95)
    if not skip_f3 and include_extras:
        print(
            "LARGE-INSTANCE BATCH: F3, Norm, New, then MD, MD Pareto, P12 (same W,D; "
            "per-solve time limit; wall time is sum of solvers)"
        )
    elif not skip_f3:
        print("LARGE-INSTANCE BATCH: F3, then Norm, then New (LARGE_BENCH_NO_EXTRA=1 — no MD/P12)")
    elif include_extras:
        print(
            "LARGE-INSTANCE BATCH: Norm, New, MD, MD Pareto, P12 (F3 SKIPPED; "
            "LARGE_BENCH_NO_EXTRA unset)"
        )
    else:
        print("LARGE-INSTANCE BATCH: Norm + New only (F3 and extras skipped)")
    print("=" * 95)

    instance_specs = get_large_instance_specs()
    seeds = [42, 43, 44]
    time_limit = 600.0

    total_runs = len(instance_specs) * len(seeds)
    print(
        f"\nInstance specs: {len(instance_specs)} configurations "
        f"(first: n=30 sanity, then n=60..200)"
    )
    print(f"Seeds per config: {seeds} -> {total_runs} total runs")
    print(f"Time limit per solve: {time_limit}s  (each solver call; not a shared cap)")
    if skip_f3:
        print("F3: skipped (unset LARGE_BENCH_SKIP_F3 to run F3)")
    if not include_extras:
        print("MD / MD Pareto / P12: skipped (unset LARGE_BENCH_NO_EXTRA to run all six solvers)")
    print("\nRunning suite...")
    print("-" * 60)

    results = run_timing_suite(
        instance_specs,
        seeds=seeds,
        time_limit=time_limit,
        use_phase1=True,
        verbose=True,
        heartbeat_sec=20.0,
        skip_f3=skip_f3,
        include_md_and_phase12=include_extras,
    )

    agg = aggregate_by_instance(results)
    print_timing_table(results, agg)
    print_detailed_results(results)

    # Summary: fastest counts
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    total = len([r for r in results if "error" not in r])
    matches = sum(1 for r in results if r.get("match"))
    f3_wins = norm_wins = new_wins = 0
    md_wins = mdp_wins = p12_wins = 0
    for r in results:
        w = _fastest_winner(r, include_extras, skip_f3)
        if w is None:
            continue
        if w == "F3":
            f3_wins += 1
        elif w == "NormB":
            norm_wins += 1
        elif w == "NewB":
            new_wins += 1
        elif w == "MD":
            md_wins += 1
        elif w == "MDP":
            mdp_wins += 1
        elif w == "P12":
            p12_wins += 1

    print(f"Total runs:             {total}")
    print(f"Objectives match:       {matches}/{total}")
    if not skip_f3:
        print(f"F3 fastest:             {f3_wins} runs")
    print(f"Normal Benders fastest: {norm_wins} runs")
    print(f"New Benders fastest:    {new_wins} runs")
    if include_extras:
        print(f"MD fastest:             {md_wins} runs")
        print(f"MD Pareto fastest:      {mdp_wins} runs")
        print(f"Pareto P12 fastest:     {p12_wins} runs")

    csv_path = os.path.join(_this_dir, "large_instance_results.csv")
    save_results_csv(results, csv_path)

    ok = matches == total
    print("\n" + ("All objectives matched." if ok else "WARNING: Some objectives did not match."))
    return ok


if __name__ == "__main__":
    ok = main()
