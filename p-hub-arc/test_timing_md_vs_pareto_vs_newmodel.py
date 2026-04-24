"""
Timing benchmark: NewModel vs MD vs MD+Pareto.

Compares wall-clock solve time for:
  1) new_model_hub_arc.solve_benders_hub_arc
  2) md_benders_hub_arc.solve_md_benders_hub_arc
  3) md_benders_hub_arc_pareto.solve_md_benders_hub_arc_pareto

Run:
    python3 test_timing_md_vs_pareto_vs_newmodel.py
"""

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from md_benders_hub_arc import solve_md_benders_hub_arc
from md_benders_hub_arc_pareto import solve_md_benders_hub_arc_pareto
from new_model_hub_arc import solve_benders_hub_arc


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
    phase1_fraction: float,
    md_pareto_method: str,
) -> Dict[str, Any]:
    new_res = solve_benders_hub_arc(
        n=n, p=p, W=W, D=D, time_limit=time_limit, verbose=False, use_phase1=True
    )
    md_res = solve_md_benders_hub_arc(
        n=n,
        p=p,
        W=W,
        D=D,
        time_limit=time_limit,
        phase1_fraction=phase1_fraction,
        verbose=False,
        use_phase1=True,
    )
    mdp_res = solve_md_benders_hub_arc_pareto(
        n=n,
        p=p,
        W=W,
        D=D,
        time_limit=time_limit,
        phase1_fraction=phase1_fraction,
        verbose=False,
        use_phase1=True,
        pareto_method=md_pareto_method,
    )

    new_obj = new_res.get("objective")
    md_obj = md_res.get("objective")
    mdp_obj = mdp_res.get("objective")
    match = (
        new_obj is not None
        and md_obj is not None
        and mdp_obj is not None
        and abs(new_obj - md_obj) < 1e-4
        and abs(new_obj - mdp_obj) < 1e-4
    )

    return {
        "n": n,
        "p": p,
        "new_time": new_res.get("time"),
        "new_obj": new_obj,
        "new_status": new_res.get("status"),
        "md_time": md_res.get("time"),
        "md_obj": md_obj,
        "md_status": md_res.get("status"),
        "mdp_time": mdp_res.get("time"),
        "mdp_obj": mdp_obj,
        "mdp_status": mdp_res.get("status"),
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
        nt = [x["new_time"] for x in rows if x.get("new_time") is not None]
        mt = [x["md_time"] for x in rows if x.get("md_time") is not None]
        pt = [x["mdp_time"] for x in rows if x.get("mdp_time") is not None]
        out[key] = {
            "n": key[0],
            "p": key[1],
            "count": len(rows),
            "new_mean": float(np.mean(nt)) if nt else None,
            "md_mean": float(np.mean(mt)) if mt else None,
            "mdp_mean": float(np.mean(pt)) if pt else None,
            "match_count": sum(1 for x in rows if x.get("match")),
        }
    return out


def main() -> bool:
    print("\n" + "=" * 124)
    print("TIMING BENCHMARK: NewModel vs MD vs MD+Pareto")
    print("=" * 124)

    specs = get_instance_specs()
    seeds = [42]
    time_limit = 300.0
    phase1_fraction = 0.3
    md_pareto_method = "two_step"

    n_fixed = sum(1 for s in specs if s.get("fixed"))
    n_random = len(specs) - n_fixed
    total_runs = n_fixed + n_random * len(seeds)
    print(f"\nConfigurations: {len(specs)}  |  Seeds: {seeds}  |  Total runs: {total_runs}")
    print(f"Time limit per solve: {time_limit}s")
    print(f"MD phase1_fraction: {phase1_fraction}")
    print(f"MD+Pareto method: {md_pareto_method}")
    print("\n" + "-" * 88)

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
                r = run_instance(
                    n=n,
                    p=p,
                    W=W,
                    D=D,
                    time_limit=time_limit,
                    phase1_fraction=phase1_fraction,
                    md_pareto_method=md_pareto_method,
                )
                r["seed"] = seed
                r["fixed"] = fixed
                results.append(r)
                ms = "Match" if r["match"] else "MISMATCH"
                print(
                    f"New={r['new_time']:.3f}s  MD={r['md_time']:.3f}s  "
                    f"MD+P={r['mdp_time']:.3f}s  | {ms}"
                )
            except Exception as e:
                print(f"ERROR: {e}")
                results.append(
                    {"n": n, "p": p, "seed": seed, "fixed": fixed, "error": str(e)}
                )

    agg = aggregate(results)
    print("\n" + "=" * 124)
    print("MEAN TIMING TABLE BY (n, p)")
    print("=" * 124)
    print(
        f"{'n':>4} | {'p':>3} | {'#runs':>6} | {'NewModel (s)':>14} | "
        f"{'MD (s)':>10} | {'MD+Pareto (s)':>15} | Match"
    )
    print("-" * 124)
    for key in sorted(agg.keys()):
        a = agg[key]
        ns = f"{a['new_mean']:.4f}" if a["new_mean"] is not None else "N/A"
        ms = f"{a['md_mean']:.4f}" if a["md_mean"] is not None else "N/A"
        ps = f"{a['mdp_mean']:.4f}" if a["mdp_mean"] is not None else "N/A"
        mc = f"{a['match_count']}/{a['count']}"
        print(
            f"{a['n']:>4} | {a['p']:>3} | {a['count']:>6} | {ns:>14} | "
            f"{ms:>10} | {ps:>15} | {mc}"
        )

    ok_rows = [r for r in results if "error" not in r]
    total = len(ok_rows)
    matches = sum(1 for r in ok_rows if r.get("match"))
    print("\n" + "=" * 124)
    print("SUMMARY")
    print("=" * 124)
    print(f"Total runs:       {total}")
    print(f"Objectives match: {matches}/{total}")
    print("\n" + ("All objectives matched." if matches == total else "WARNING: Some objectives did not match."))
    return matches == total


if __name__ == "__main__":
    main()

