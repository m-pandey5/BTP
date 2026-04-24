"""
New Pareto-Benders variant for p-Hub-Arc.

This file provides a stable "new" entry point for benchmarking:
  - Core point can blend uniform MW core with Phase-1 LP solution.
  - Blend is controlled by core_lp_blend in [0, 1].
"""

from typing import Any, Dict, List

from pareto_benders_hub_arc import solve_benders_pareto_hub_arc as _solve_base


def solve_benders_pareto_hub_arc_new(
    n: int,
    p: int,
    W: List[List[float]],
    D: List[List[float]],
    time_limit: float = None,
    verbose: bool = False,
    use_phase1: bool = True,
    core_lp_blend: float = 0.5,
) -> Dict[str, Any]:
    """
    Solve p-hub-arc with the new Pareto strategy (LP-blended MW core).

    Notes
    -----
    - core_lp_blend=0.0 reproduces the old behavior.
    - Suggested benchmark values: 0.2, 0.35, 0.5.
    """
    return _solve_base(
        n=n,
        p=p,
        W=W,
        D=D,
        time_limit=time_limit,
        verbose=verbose,
        use_phase1=use_phase1,
        core_lp_blend=core_lp_blend,
        pareto_method="two_step",
    )

