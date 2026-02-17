"""
Comprehensive Timing Test: F3 vs Benders for p-Hub-Arc

Compares:
  - F3 (direct canonical formulation)
  - Normal Benders (HubArcBenders: master + lazy callback, no Phase 1)
  - New Benders (solve_benders_hub_arc: Phase 1 LP + Phase 2 with warm start)

Run: python test_timing_comprehensive.py
"""

import sys
import os
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from hub_arc_models import solve_hub_arc_F3, HubArcBenders
from new_model_hub_arc import solve_benders_hub_arc, preprocess


def build_instance(n: int, p: int, seed: Optional[int] = None) -> Tuple[List, List]:
    """Build random W (flows) and D (distances) for hub-arc."""
    if seed is not None:
        np.random.seed(seed)
    W = np.random.rand(n, n) * 10
    np.fill_diagonal(W, 0)
    D = np.random.rand(n, n) * 20
    np.fill_diagonal(D, 0)
    return W.tolist(), D.tolist()


def run_instance(
    n: int,
    p: int,
    W: List,
    D: List,
    time_limit: Optional[float] = None,
    use_phase1: bool = True,
) -> Dict[str, Any]:
    """
    Run F3, normal Benders (HubArcBenders), and new Benders on one instance.
    Returns dict with objectives, times, status, and match.
    """
    f3_res = solve_hub_arc_F3(n, p, W, D, gurobi_output=False)

    # Normal Benders (HubArcBenders): no Phase 1, callback-only
    b_norm = HubArcBenders(n=n, p=p, W=W, D=D, verbose=False)
    norm_res = b_norm.solve(time_limit=time_limit)
    # HubArcBenders returns master.ObjVal; add constant for OD pairs with K==1
    _, C, _, K, _, _, _ = preprocess(n, W, D)
    if norm_res["objective"] is not None:
        add_const = sum(C[(i, j)][0] for (i, j) in K if i != j and K[(i, j)] == 1)
        norm_res = dict(norm_res, objective=norm_res["objective"] + add_const)

    # New Benders (Phase 1 + Phase 2)
    new_res = solve_benders_hub_arc(
        n, p, W, D, verbose=False, use_phase1=use_phase1, time_limit=time_limit
    )

    ref = f3_res["objective"]
    match_f3 = ref is not None
    diff_norm = diff_new = None
    if ref is not None:
        if norm_res["objective"] is not None:
            diff_norm = abs(ref - norm_res["objective"])
            match_f3 = match_f3 and diff_norm < 1e-4
        if new_res["objective"] is not None:
            diff_new = abs(ref - new_res["objective"])
            match_f3 = match_f3 and diff_new < 1e-4

    return {
        "n": n,
        "p": p,
        "f3_obj": ref,
        "f3_time": f3_res["time"],
        "f3_status": f3_res["status"],
        "norm_obj": norm_res["objective"],
        "norm_time": norm_res["time"],
        "norm_status": norm_res["status"],
        "new_obj": new_res["objective"],
        "new_time": new_res["time"],
        "new_status": new_res["status"],
        "match": match_f3,
        "diff_norm": diff_norm,
        "diff_new": diff_new,
    }


def run_timing_suite(
    instance_specs: List[Dict],
    seeds: List[int],
    time_limit: Optional[float] = 300.0,
    use_phase1: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run full timing suite: each (n, p) × each seed.
    Returns list of results with aggregated stats per (n, p).
    """
    results = []
    idx = 0

    for spec in instance_specs:
        n = spec["n"]
        p = spec["p"]
        fixed = spec.get("fixed", False)
        seed_list = [None] if fixed else seeds

        for seed in seed_list:
            idx += 1
            if verbose:
                label = f"n={n}, p={p}" + (f", seed={seed}" if seed is not None else " (fixed)")
                print(f"  [{idx}] {label} ...", end=" ", flush=True)

            if fixed:
                W, D = spec["W"], spec["D"]
            else:
                W, D = build_instance(n, p, seed)

            try:
                r = run_instance(n, p, W, D, time_limit=time_limit, use_phase1=use_phase1)
                r["seed"] = seed
                r["fixed"] = fixed
                results.append(r)
                if verbose:
                    f3t, nt, nn = r["f3_time"], r["norm_time"], r["new_time"]
                    times = [("F3", f3t), ("Norm", nt), ("New", nn)]
                    fastest = min(times, key=lambda x: x[1] or float("inf"))
                    print(f"F3={f3t:.3f}s, NormB={nt:.3f}s, NewB={nn:.3f}s (fastest: {fastest[0]})")
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                results.append({
                    "n": n, "p": p, "seed": seed, "fixed": fixed,
                    "f3_time": None, "norm_time": None, "new_time": None,
                    "match": False, "error": str(e),
                })

    return results


def aggregate_by_instance(results: List[Dict]) -> Dict[Tuple[int, int], Dict]:
    """Aggregate results by (n, p): mean time, std, count, match rate."""
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        key = (r["n"], r["p"])
        by_key[key].append(r)

    agg = {}
    for (n, p), rows in by_key.items():
        f3_t = [x["f3_time"] for x in rows if x.get("f3_time") is not None]
        norm_t = [x["norm_time"] for x in rows if x.get("norm_time") is not None]
        new_t = [x["new_time"] for x in rows if x.get("new_time") is not None]
        matches = sum(1 for x in rows if x.get("match"))
        agg[(n, p)] = {
            "n": n,
            "p": p,
            "count": len(rows),
            "f3_mean": np.mean(f3_t) if f3_t else None,
            "norm_mean": np.mean(norm_t) if norm_t else None,
            "new_mean": np.mean(new_t) if new_t else None,
            "match_count": matches,
            "match_rate": matches / len(rows) if rows else 0,
        }
    return agg


def print_timing_table(results: List[Dict], agg: Dict):
    """Print formatted timing comparison table."""
    print("\n" + "=" * 115)
    print("TIMING COMPARISON TABLE (F3 vs Normal Benders vs New Benders)")
    print("=" * 115)

    header = f"{'n':>4} | {'p':>3} | {'#runs':>6} | {'F3 (s)':>10} | {'Norm Benders (s)':>16} | {'New Benders (s)':>16} | {'Fastest':>12} | Match"
    print(header)
    print("-" * 115)

    for key in sorted(agg.keys(), key=lambda k: (k[0], k[1])):
        a = agg[key]
        n, p = a["n"], a["p"]
        fm = a["f3_mean"]
        nm = a["norm_mean"]
        newm = a["new_mean"]
        count = a["count"]

        f_str = f"{fm:.4f}" if fm is not None else "   N/A   "
        n_str = f"{nm:.4f}" if nm is not None else "   N/A   "
        new_str = f"{newm:.4f}" if newm is not None else "   N/A   "

        times = [("F3", fm), ("NormB", nm), ("NewB", newm)]
        valid = [(name, t) for name, t in times if t is not None and t >= 0]
        fastest = min(valid, key=lambda x: x[1])[0] if valid else "-"
        match_str = f"{a['match_count']}/{count}"
        print(f"{n:>4} | {p:>3} | {count:>6} | {f_str:>10} | {n_str:>16} | {new_str:>16} | {fastest:>12} | {match_str}")


def print_detailed_results(results: List[Dict]):
    """Print per-run details."""
    print("\n" + "=" * 115)
    print("DETAILED PER-RUN RESULTS")
    print("=" * 115)
    for i, r in enumerate(results):
        if "error" in r:
            print(f"  [{i+1}] n={r['n']}, p={r['p']}: ERROR - {r['error']}")
            continue
        status = "OK" if r.get("match") else "MISMATCH"
        f3t = r.get("f3_time")
        nt = r.get("norm_time")
        nn = r.get("new_time")
        f3s = f"{f3t:.4f}" if f3t is not None else "N/A"
        ns = f"{nt:.4f}" if nt is not None else "N/A"
        nns = f"{nn:.4f}" if nn is not None else "N/A"
        print(f"  [{i+1}] n={r['n']}, p={r['p']}, seed={r.get('seed')}: "
              f"F3={f3s}s, NormB={ns}s, NewB={nns}s [{status}]")


def main():
    print("\n" + "=" * 95)
    print("COMPREHENSIVE TIMING TEST: p-Hub-Arc (F3 vs Normal Benders vs New Benders)")
    print("=" * 95)

    # Fixed instances (small, deterministic)
    W1 = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D1 = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    W2 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D2 = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]

    instance_specs = [
        {"n": 3, "p": 2, "fixed": True, "W": W1, "D": D1},
        {"n": 3, "p": 3, "fixed": True, "W": W1, "D": D1},
        {"n": 4, "p": 2, "fixed": True, "W": W2, "D": D2},
        {"n": 4, "p": 3, "fixed": True, "W": W2, "D": D2},
        {"n": 5, "p": 2, "fixed": False},
        {"n": 5, "p": 3, "fixed": False},
        {"n": 6, "p": 3, "fixed": False},
        {"n": 6, "p": 4, "fixed": False},
        {"n": 7, "p": 4, "fixed": False},
        {"n": 8, "p": 4, "fixed": False},
        {"n": 8, "p": 5, "fixed": False},
        {"n": 10, "p": 5, "fixed": False},
        {"n": 10, "p": 6, "fixed": False},
    ]

    seeds = [42, 43, 44, 45, 46]
    time_limit = 120.0

    print(f"\nInstance specs: {len(instance_specs)} configurations")
    print(f"Random seeds per config: {len(seeds)} (or 1 for fixed)")
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
        valid = [(n, t) for n, t in times if t is not None]
        if valid:
            fastest = min(valid, key=lambda x: x[1])[0]
            if fastest == "F3":
                f3_wins += 1
            elif fastest == "NormB":
                norm_wins += 1
            else:
                new_wins += 1

    print(f"Total runs:       {total}")
    print(f"Objectives match: {matches}/{total}")
    print(f"F3 fastest:       {f3_wins} runs")
    print(f"Normal Benders fastest: {norm_wins} runs")
    print(f"New Benders fastest:    {new_wins} runs")

    ok = matches == total
    print("\n" + ("All objectives matched." if ok else "WARNING: Some objectives did not match."))
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
