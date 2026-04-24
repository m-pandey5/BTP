"""
Timing benchmark: F3 vs New Benders vs Pareto-Old vs Pareto-New.

This script is designed for the same "progress line + time" style as
test_pareto_benders.py so you can compare formulations directly.

Run:
    python3 test_timing_pareto_old_new.py
"""

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from hub_arc_models import solve_hub_arc_F3
from new_model_hub_arc import solve_benders_hub_arc
from pareto_benders_hub_arc_old import solve_benders_pareto_hub_arc_old
from pareto_benders_hub_arc_new import solve_benders_pareto_hub_arc_new
from test_pareto_benders import build_instance, get_instance_specs


def run_instance(
    n: int,
    p: int,
    W: List,
    D: List,
    time_limit: Optional[float] = None,
    use_phase1: bool = True,
    core_lp_blend: float = 0.35,
) -> Dict[str, Any]:
    """Run all formulations and return objectives/times/statuses."""
    f3_res = solve_hub_arc_F3(n, p, W, D, gurobi_output=False)
    new_res = solve_benders_hub_arc(
        n, p, W, D, verbose=False, use_phase1=use_phase1, time_limit=time_limit
    )
    old_res = solve_benders_pareto_hub_arc_old(
        n, p, W, D, verbose=False, use_phase1=use_phase1, time_limit=time_limit
    )
    newp_res = solve_benders_pareto_hub_arc_new(
        n,
        p,
        W,
        D,
        verbose=False,
        use_phase1=use_phase1,
        time_limit=time_limit,
        core_lp_blend=core_lp_blend,
    )

    ref = f3_res["objective"]
    match_all = ref is not None
    diffs = {"new": None, "old": None, "newp": None}

    if ref is not None:
        if new_res["objective"] is not None:
            diffs["new"] = abs(ref - new_res["objective"])
            match_all = match_all and diffs["new"] < 1e-4
        if old_res["objective"] is not None:
            diffs["old"] = abs(ref - old_res["objective"])
            match_all = match_all and diffs["old"] < 1e-4
        if newp_res["objective"] is not None:
            diffs["newp"] = abs(ref - newp_res["objective"])
            match_all = match_all and diffs["newp"] < 1e-4

    return {
        "n": n,
        "p": p,
        "f3_obj": f3_res["objective"],
        "f3_time": f3_res["time"],
        "f3_status": f3_res["status"],
        "new_obj": new_res["objective"],
        "new_time": new_res["time"],
        "new_status": new_res["status"],
        "old_obj": old_res["objective"],
        "old_time": old_res["time"],
        "old_status": old_res["status"],
        "newp_obj": newp_res["objective"],
        "newp_time": newp_res["time"],
        "newp_status": newp_res["status"],
        "match": match_all,
        "diff_new": diffs["new"],
        "diff_old": diffs["old"],
        "diff_newp": diffs["newp"],
    }


def aggregate_by_instance(results: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """Aggregate mean times by (n, p)."""
    by_key: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        by_key[(r["n"], r["p"])].append(r)

    agg = {}
    for (n, p), rows in by_key.items():
        agg[(n, p)] = {
            "n": n,
            "p": p,
            "count": len(rows),
            "f3_mean": np.mean([x["f3_time"] for x in rows if x.get("f3_time") is not None]),
            "new_mean": np.mean([x["new_time"] for x in rows if x.get("new_time") is not None]),
            "old_mean": np.mean([x["old_time"] for x in rows if x.get("old_time") is not None]),
            "newp_mean": np.mean([x["newp_time"] for x in rows if x.get("newp_time") is not None]),
            "match_count": sum(1 for x in rows if x.get("match")),
        }
    return agg


def main() -> bool:
    print("\n" + "=" * 120)
    print("TIMING BENCHMARK: F3 vs New Benders vs Pareto-Old vs Pareto-New")
    print("=" * 120)

    specs = get_instance_specs()
    seeds = [42]
    time_limit = 300.0
    core_lp_blend = 0.35

    n_fixed = sum(1 for s in specs if s.get("fixed"))
    n_random = len(specs) - n_fixed
    total_runs = n_fixed + n_random * len(seeds)

    print(
        f"\nConfigurations: {len(specs)}  |  Seeds: {seeds}  |  Total runs: {total_runs}"
    )
    print(f"Time limit per solve: {time_limit}s")
    print(f"Pareto-New core_lp_blend: {core_lp_blend}")
    print("\n" + "-" * 80)

    results: List[Dict[str, Any]] = []
    idx = 0
    for spec in specs:
        n, p = spec["n"], spec["p"]
        fixed = spec.get("fixed", False)
        seed_list = [None] if fixed else seeds

        for seed in seed_list:
            idx += 1
            label = f"n={n}, p={p}" + (f", seed={seed}" if seed is not None else " (fixed)")
            print(f"  [{idx}] {label} ...", end=" ", flush=True)

            W, D = (spec["W"], spec["D"]) if fixed else build_instance(n, p, seed)

            try:
                r = run_instance(
                    n=n,
                    p=p,
                    W=W,
                    D=D,
                    time_limit=time_limit,
                    use_phase1=True,
                    core_lp_blend=core_lp_blend,
                )
                r["seed"] = seed
                r["fixed"] = fixed
                results.append(r)

                f3_t = r["f3_time"]
                new_t = r["new_time"]
                old_t = r["old_time"]
                newp_t = r["newp_time"]
                match_str = "Match" if r["match"] else "MISMATCH"
                print(
                    f"F3={f3_t:.3f}s  New={new_t:.3f}s  P-Old={old_t:.3f}s  "
                    f"P-New={newp_t:.3f}s  | {match_str}"
                )
            except Exception as e:
                print(f"ERROR: {e}")
                results.append(
                    {"n": n, "p": p, "seed": seed, "fixed": fixed, "match": False, "error": str(e)}
                )

    agg = aggregate_by_instance(results)

    print("\n" + "=" * 120)
    print("MEAN TIMING TABLE BY (n, p)")
    print("=" * 120)
    print(
        f"{'n':>4} | {'p':>3} | {'#runs':>6} | {'F3 (s)':>10} | {'New (s)':>10} | "
        f"{'P-Old (s)':>12} | {'P-New (s)':>12} | Match"
    )
    print("-" * 120)
    for key in sorted(agg.keys()):
        a = agg[key]
        m = f"{a['match_count']}/{a['count']}"
        print(
            f"{a['n']:>4} | {a['p']:>3} | {a['count']:>6} | "
            f"{a['f3_mean']:>10.4f} | {a['new_mean']:>10.4f} | "
            f"{a['old_mean']:>12.4f} | {a['newp_mean']:>12.4f} | {m}"
        )

    ok_rows = [r for r in results if "error" not in r]
    total = len(ok_rows)
    matches = sum(1 for r in ok_rows if r.get("match"))
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"Total runs:       {total}")
    print(f"Objectives match: {matches}/{total}")
    print("\n" + ("All objectives matched." if matches == total else "WARNING: Some objectives did not match."))
    return matches == total


if __name__ == "__main__":
    main()

