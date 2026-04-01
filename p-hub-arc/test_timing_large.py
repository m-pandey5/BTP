"""
Large-Instance Batch Timing Test: F3 vs Normal Benders vs New Benders

Runs all three solvers on large random instances (n=60..200) across multiple seeds.
Results are printed as a timing table and saved to large_instance_results.csv.

Run: python p-hub-arc/test_timing_large.py
"""

import csv
import os
import sys
from typing import Dict, List

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from test_timing_comprehensive import (
    aggregate_by_instance,
    build_instance,
    print_detailed_results,
    print_timing_table,
    run_instance,
    run_timing_suite,
)


def get_large_instance_specs() -> List[Dict]:
    """Large random instance configs: n=60..200, two p values each."""
    configs = []
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
    """Save per-run results to a CSV file."""
    fieldnames = [
        "n", "p", "seed",
        "f3_time", "f3_obj", "f3_status",
        "norm_time", "norm_obj", "norm_status",
        "new_time", "new_obj", "new_status",
        "match",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"\nResults saved to: {path}")


def main():
    print("\n" + "=" * 95)
    print("LARGE-INSTANCE BATCH TIMING TEST: p-Hub-Arc (F3 vs Normal Benders vs New Benders)")
    print("=" * 95)

    instance_specs = get_large_instance_specs()
    seeds = [42, 43, 44]
    time_limit = 600.0

    total_runs = len(instance_specs) * len(seeds)
    print(f"\nInstance specs: {len(instance_specs)} configurations (n=60..200)")
    print(f"Seeds per config: {seeds} -> {total_runs} total runs")
    print(f"Time limit per solve: {time_limit}s")
    print("\nRunning suite...")
    print("-" * 60)

    results = run_timing_suite(
        instance_specs,
        seeds=seeds,
        time_limit=time_limit,
        use_phase1=True,
        verbose=True,
    )

    agg = aggregate_by_instance(results)
    print_timing_table(results, agg)
    print_detailed_results(results)

    # Summary
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    total = len([r for r in results if "error" not in r])
    matches = sum(1 for r in results if r.get("match"))
    f3_wins = norm_wins = new_wins = 0
    for r in results:
        if "error" in r:
            continue
        times = [("F3", r.get("f3_time")), ("NormB", r.get("norm_time")), ("NewB", r.get("new_time"))]
        valid = [(name, t) for name, t in times if t is not None]
        if valid:
            fastest = min(valid, key=lambda x: x[1])[0]
            if fastest == "F3":
                f3_wins += 1
            elif fastest == "NormB":
                norm_wins += 1
            else:
                new_wins += 1

    print(f"Total runs:             {total}")
    print(f"Objectives match:       {matches}/{total}")
    print(f"F3 fastest:             {f3_wins} runs")
    print(f"Normal Benders fastest: {norm_wins} runs")
    print(f"New Benders fastest:    {new_wins} runs")

    csv_path = os.path.join(_this_dir, "large_instance_results.csv")
    save_results_csv(results, csv_path)

    ok = matches == total
    print("\n" + ("All objectives matched." if ok else "WARNING: Some objectives did not match."))
    return ok


if __name__ == "__main__":
    ok = main()
