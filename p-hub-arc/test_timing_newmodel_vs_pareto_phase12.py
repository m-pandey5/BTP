"""
Timing comparison: new_model_hub_arc vs Pareto cuts in Phase 1+2.

Run:
    python3 test_timing_newmodel_vs_pareto_phase12.py
"""

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from new_model_hub_arc import solve_benders_hub_arc
from new_model_hub_arc_pareto_phase12 import solve_benders_hub_arc_pareto_phase12


def build_instance(n: int, seed: Optional[int] = None) -> Tuple[List, List]:
    if seed is not None:
        np.random.seed(seed)
    W = np.random.rand(n, n) * 10
    np.fill_diagonal(W, 0)
    D = np.random.rand(n, n) * 20
    np.fill_diagonal(D, 0)
    return W.tolist(), D.tolist()


def get_instance_specs() -> List[Dict[str, Any]]:
    W1 = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D1 = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    W2 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D2 = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    return [
        {"n": 3, "p": 2, "fixed": True, "W": W1, "D": D1},
        {"n": 4, "p": 2, "fixed": True, "W": W2, "D": D2},
        {"n": 5, "p": 2, "fixed": False},
        {"n": 6, "p": 3, "fixed": False},
        {"n": 8, "p": 4, "fixed": False},
        {"n": 10, "p": 4, "fixed": False},
        {"n": 12, "p": 5, "fixed": False},
        {"n": 15, "p": 6, "fixed": False},
        {"n": 20, "p": 8, "fixed": False},
    ]


def run_instance(
    n: int,
    p: int,
    W: List,
    D: List,
    time_limit: float,
    pareto_method: str,
) -> Dict[str, Any]:
    base = solve_benders_hub_arc(
        n=n, p=p, W=W, D=D, time_limit=time_limit, verbose=False, use_phase1=True
    )
    p12 = solve_benders_hub_arc_pareto_phase12(
        n=n,
        p=p,
        W=W,
        D=D,
        time_limit=time_limit,
        verbose=False,
        use_phase1=True,
        pareto_method=pareto_method,
    )

    base_obj = base.get("objective")
    p12_obj = p12.get("objective")
    match = False
    if base_obj is not None and p12_obj is not None:
        match = abs(base_obj - p12_obj) < 1e-4

    return {
        "n": n,
        "p": p,
        "base_time": base.get("time"),
        "base_obj": base_obj,
        "base_status": base.get("status"),
        "p12_time": p12.get("time"),
        "p12_obj": p12_obj,
        "p12_status": p12.get("status"),
        "match": match,
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    by_key: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        by_key[(r["n"], r["p"])].append(r)

    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for key, rows in by_key.items():
        bt = [x["base_time"] for x in rows if x.get("base_time") is not None]
        pt = [x["p12_time"] for x in rows if x.get("p12_time") is not None]
        out[key] = {
            "n": key[0],
            "p": key[1],
            "count": len(rows),
            "base_mean": float(np.mean(bt)) if bt else None,
            "p12_mean": float(np.mean(pt)) if pt else None,
            "match_count": sum(1 for x in rows if x.get("match")),
        }
    return out


def main() -> bool:
    print("\n" + "=" * 116)
    print("TIMING: new_model_hub_arc vs Pareto Phase1+Phase2")
    print("=" * 116)

    specs = get_instance_specs()
    seeds = [42]
    time_limit = 300.0
    pareto_method = "two_step"

    n_fixed = sum(1 for s in specs if s.get("fixed"))
    n_random = len(specs) - n_fixed
    total_runs = n_fixed + n_random * len(seeds)

    print(f"\nConfigurations: {len(specs)}  |  Seeds: {seeds}  |  Total runs: {total_runs}")
    print(f"Time limit per solve: {time_limit}s")
    print(f"Pareto method: {pareto_method}")
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

            W, D = (spec["W"], spec["D"]) if fixed else build_instance(n, seed)
            try:
                r = run_instance(n, p, W, D, time_limit, pareto_method)
                r["seed"] = seed
                r["fixed"] = fixed
                results.append(r)
                ms = "Match" if r["match"] else "MISMATCH"
                print(
                    f"Base={r['base_time']:.3f}s  ParetoPh12={r['p12_time']:.3f}s  | {ms}"
                )
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"n": n, "p": p, "seed": seed, "fixed": fixed, "error": str(e)})

    agg = aggregate(results)
    print("\n" + "=" * 116)
    print("MEAN TIMING TABLE")
    print("=" * 116)
    print(f"{'n':>4} | {'p':>3} | {'#runs':>6} | {'Base NewModel (s)':>18} | {'Pareto Ph1+Ph2 (s)':>20} | Match")
    print("-" * 116)
    for key in sorted(agg.keys()):
        a = agg[key]
        b = f"{a['base_mean']:.4f}" if a["base_mean"] is not None else "N/A"
        p12 = f"{a['p12_mean']:.4f}" if a["p12_mean"] is not None else "N/A"
        m = f"{a['match_count']}/{a['count']}"
        print(f"{a['n']:>4} | {a['p']:>3} | {a['count']:>6} | {b:>18} | {p12:>20} | {m}")

    ok_rows = [r for r in results if "error" not in r]
    total = len(ok_rows)
    matches = sum(1 for r in ok_rows if r.get("match"))
    print("\n" + "=" * 116)
    print("SUMMARY")
    print("=" * 116)
    print(f"Total runs:       {total}")
    print(f"Objectives match: {matches}/{total}")
    print("\n" + ("All objectives matched." if matches == total else "WARNING: Some objectives did not match."))
    return matches == total


if __name__ == "__main__":
    main()

