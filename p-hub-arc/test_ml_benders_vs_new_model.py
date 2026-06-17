"""
Compare ML-assisted Benders (ml_benders_hub_arc) vs standard New Benders (new_model_hub_arc).

Run from p-hub-arc:
    python3 test_ml_benders_vs_new_model.py

Uses two ML configurations:
  1) exact_compare — all OD pairs checked, no cut cap (should match New Benders obj)
  2) default_ml    — 25% OD prioritization + cut cap (may differ in path/time, obj should still match if optimal)
"""

import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

import numpy as np

from new_model_hub_arc import solve_benders_hub_arc
from ml_benders_hub_arc import solve_ml_benders_hub_arc
from test_timing_comprehensive import build_instance

TOL = 1e-4
TIME_LIMIT = 120.0


def _fixed_4x4():
  W = [
      [0, 10, 20, 15],
      [12, 0, 18, 25],
      [14, 11, 0, 30],
      [20, 16, 22, 0],
  ]
  D = [
      [0, 2, 6, 8],
      [2, 0, 4, 5],
      [6, 4, 0, 3],
      [8, 5, 3, 0],
  ]
  return W, D


def compare_case(n: int, p: int, W, D, seed_label: str) -> bool:
    print(f"\n--- {seed_label} n={n} p={p} ---")

    base = solve_benders_hub_arc(
        n, p, W, D, time_limit=TIME_LIMIT, verbose=False, use_phase1=True
    )
    ml_exact = solve_ml_benders_hub_arc(
        n,
        p,
        W,
        D,
        time_limit=TIME_LIMIT,
        verbose=False,
        use_phase1=True,
        arc_model=None,
        od_model=None,
        ml_pair_fraction=1.0,
        max_cuts_per_callback=None,
    )
    ml_default = solve_ml_benders_hub_arc(
        n,
        p,
        W,
        D,
        time_limit=TIME_LIMIT,
        verbose=False,
        use_phase1=True,
        arc_model=None,
        od_model=None,
        ml_pair_fraction=0.25,
        max_cuts_per_callback=20,
    )

    print(
        f"  New Benders:  obj={base['objective']:.6f}  "
        f"status={base['status']}  time={base['time']:.3f}s"
    )
    print(
        f"  ML (exact):   obj={ml_exact['objective']:.6f}  "
        f"status={ml_exact['status']}  time={ml_exact['time']:.3f}s  "
        f"fallbacks={ml_exact.get('full_fallback_calls', 0)}"
    )
    print(
        f"  ML (default): obj={ml_default['objective']:.6f}  "
        f"status={ml_default['status']}  time={ml_default['time']:.3f}s  "
        f"fallbacks={ml_default.get('full_fallback_calls', 0)}"
    )

    ok_base = base["objective"] is not None and base["status"] == "OPTIMAL"
    ok_ml_exact = ml_exact["objective"] is not None and ml_exact["status"] == "OPTIMAL"
    ok_ml_default = ml_default["objective"] is not None and ml_default["status"] == "OPTIMAL"

    if not ok_base:
        print("  FAIL: New Benders did not reach OPTIMAL")
        return False

    match_exact = (
        ok_ml_exact
        and abs(base["objective"] - ml_exact["objective"]) < TOL
    )
    match_default = (
        ok_ml_default
        and abs(base["objective"] - ml_default["objective"]) < TOL
    )

    print(f"  ML exact matches New Benders:   {'YES' if match_exact else 'NO'}")
    print(f"  ML default matches New Benders: {'YES' if match_default else 'NO'}")

    return match_exact and match_default


def main():
    cases = []

    W, D = _fixed_4x4()
    cases.append((4, 2, W, D, "fixed 4x4"))

    for n, p, seed in [(4, 2, 42), (5, 3, 7), (6, 4, 99)]:
        W, D = build_instance(n, p, seed)
        cases.append((n, p, W, D, f"random seed={seed}"))

    passed = 0
    for n, p, W, D, label in cases:
        if compare_case(n, p, W, D, label):
            passed += 1

    total = len(cases)
    print(f"\n{'=' * 60}")
    print(f"Passed {passed}/{total} comparison cases (tol={TOL})")
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
