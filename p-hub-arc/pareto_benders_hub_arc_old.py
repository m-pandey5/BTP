"""
Old Pareto-Benders variant for p-Hub-Arc.

This file provides a stable "old" entry point for benchmarking:
  - Core point is always uniform: y_core[a] = p / |H|
  - No LP-core blending is used.

Implementation delegates to the main solver with core_lp_blend=0.0.
"""

from typing import Any, Dict, List

from pareto_benders_hub_arc import solve_benders_pareto_hub_arc as _solve_base


def solve_benders_pareto_hub_arc_old(
    n: int,
    p: int,
    W: List[List[float]],
    D: List[List[float]],
    time_limit: float = None,
    verbose: bool = False,
    use_phase1: bool = True,
) -> Dict[str, Any]:
    """
    Solve p-hub-arc with the old Pareto strategy (uniform MW core).
    """
    return _solve_base(
        n=n,
        p=p,
        W=W,
        D=D,
        time_limit=time_limit,
        verbose=verbose,
        use_phase1=use_phase1,
        core_lp_blend=0.0,
        pareto_method="epsilon",
    )

