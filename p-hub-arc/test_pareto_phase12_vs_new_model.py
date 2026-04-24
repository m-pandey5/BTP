"""
Test: Pareto cuts in both phases vs standard new_model_hub_arc solver.

Compares:
  - solve_benders_hub_arc (standard: regular cuts in Phase 1 + Phase 2)
  - solve_benders_hub_arc_pareto_phase12 (Pareto cuts in Phase 1 + Phase 2)

Run:
    python -m pytest p-hub-arc/test_pareto_phase12_vs_new_model.py -v
    python p-hub-arc/test_pareto_phase12_vs_new_model.py
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

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


def run_compare(
    n: int,
    p: int,
    W: List,
    D: List,
    time_limit: float = 120.0,
    pareto_method: str = "two_step",
) -> Dict:
    base = solve_benders_hub_arc(
        n=n, p=p, W=W, D=D, time_limit=time_limit, verbose=False, use_phase1=True
    )
    par12 = solve_benders_hub_arc_pareto_phase12(
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
    par_obj = par12.get("objective")
    diff = None if (base_obj is None or par_obj is None) else abs(base_obj - par_obj)

    return {
        "n": n,
        "p": p,
        "base": base,
        "par12": par12,
        "diff": diff,
        "match": diff is not None and diff < 1e-4,
    }


def test_phase12_pareto_returns_expected_keys():
    W, D = build_instance(5, seed=11)
    res = solve_benders_hub_arc_pareto_phase12(5, 2, W, D, time_limit=60.0)
    for key in ("objective", "selected_arcs", "time", "status"):
        assert key in res


def test_phase12_vs_base_fixed_n3_p2():
    W = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    r = run_compare(3, 2, W, D, time_limit=60.0)
    assert r["match"], f"base={r['base']['objective']} par12={r['par12']['objective']}"


def test_phase12_vs_base_fixed_n4_p2():
    W = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    r = run_compare(4, 2, W, D, time_limit=60.0)
    assert r["match"], f"base={r['base']['objective']} par12={r['par12']['objective']}"


def test_phase12_vs_base_random_n6_p3_seed42():
    W, D = build_instance(6, seed=42)
    r = run_compare(6, 3, W, D, time_limit=120.0)
    assert r["match"], f"base={r['base']['objective']} par12={r['par12']['objective']}"


def test_phase12_vs_base_random_n8_p4_seed44():
    W, D = build_instance(8, seed=44)
    r = run_compare(8, 4, W, D, time_limit=180.0)
    assert r["match"], f"base={r['base']['objective']} par12={r['par12']['objective']}"


def test_phase12_selected_arcs_count_when_optimal():
    W, D = build_instance(7, seed=50)
    p = 4
    r = solve_benders_hub_arc_pareto_phase12(7, p, W, D, time_limit=120.0)
    if r["status"] == "OPTIMAL":
        assert len(r["selected_arcs"]) == p


def main() -> bool:
    print("\n" + "=" * 100)
    print("PHASE1+PHASE2 PARETO TEST: vs new_model_hub_arc")
    print("=" * 100)

    configs = [
        (3, 2, None),
        (4, 2, None),
        (6, 3, 42),
        (8, 4, 44),
        (10, 4, 52),
    ]
    ok = True
    for idx, (n, p, seed) in enumerate(configs, start=1):
        if seed is None:
            if n == 3:
                W = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
                D = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
            else:
                W = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
                D = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
        else:
            W, D = build_instance(n, seed=seed)

        label = f"n={n}, p={p}" + (f", seed={seed}" if seed is not None else " (fixed)")
        print(f"  [{idx}] {label} ...", end=" ", flush=True)
        r = run_compare(n, p, W, D, time_limit=240.0)
        b = r["base"]
        p2 = r["par12"]
        bs = "N/A" if b["objective"] is None else f"{b['objective']:.4f}"
        ps = "N/A" if p2["objective"] is None else f"{p2['objective']:.4f}"
        ms = "Match" if r["match"] else "MISMATCH"
        print(
            f"Base={b['time']:.3f}s  ParetoPh12={p2['time']:.3f}s  | {ms}"
        )
        print(f"       Obj: Base={bs}  ParetoPh12={ps}")
        ok = ok and r["match"]

    print("\n" + ("All compared objectives matched." if ok else "WARNING: Some objectives did not match."))
    return ok


if __name__ == "__main__":
    main()

