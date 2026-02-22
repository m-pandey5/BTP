"""
Algorithm 1: Separation algorithm (Section 3.2 Separation Problem)

Generates violated Benders cuts (equation 20) and computes upper bound.
- separation_algorithm_dual: solves dual subproblem (correct, used by default)
- separation_algorithm: polynomial separation using Algorithm 2 (faster but had bugs)

Reference: Section 3.3 Polynomial separation algorithm from the paper.
"""

import gurobipy as gp
from gurobipy import GRB

from algo2 import compute_k_i


def separation_algorithm_dual(y_bar, theta_bar, facilities_at_distance, D, K, N, M, tol=1e-6):
    """
    Dual-based separation: solve the dual subproblem for each client.
    Returns correct Benders cuts (same structure as polynomial version).
    """
    cuts_to_add = []
    UB = 0.0

    for i in range(N):
        sp_obj, cut = _solve_dual_subproblem(i, y_bar, facilities_at_distance, D, K)
        if sp_obj is None:
            continue
        UB += sp_obj
        if theta_bar[i] < sp_obj - tol:
            cuts_to_add.append((i, {
                "D1_i": cut["D1_i"],
                "nu": cut["nu_k"],
                "facilities_at_D": {k: cut.get(f"facilities_at_D{k}", []) for k in range(1, K[i] + 1)},
                "k_i": None,
            }))

    return cuts_to_add, UB


def _solve_dual_subproblem(i, y_bar, facilities_at_distance, D, K):
    """Solve dual subproblem for client i; return (obj, cut_data)."""
    model = gp.Model(f"DSP_{i}")
    model.Params.OutputFlag = 0

    Ki = K[i]
    nu = model.addVars(range(1, Ki + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="nu")
    fac_D = facilities_at_distance[i]

    obj_expr = nu[1] * (1.0 - sum(y_bar[j] for j in fac_D[1]))
    for k in range(2, Ki + 1):
        facs = fac_D.get(k, [])
        obj_expr += nu[k] * (-sum(y_bar[j] for j in facs))

    model.setObjective(obj_expr, GRB.MAXIMIZE)
    model.addConstr(nu[1] <= D[i][2] - D[i][1], name="c1")
    for k in range(2, Ki):
        model.addConstr(nu[k] - nu[k + 1] <= D[i][k + 1] - D[i][k], name=f"c{k}")
    if Ki > 1:
        model.addConstr(nu[Ki] <= 0, name=f"c{Ki}")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        nu_sol = {k: nu[k].X for k in range(1, Ki + 1)}
        cut = {"D1_i": D[i][1], "nu_k": nu_sol}
        for k in range(1, Ki + 1):
            cut[f"facilities_at_D{k}"] = fac_D.get(k, [])
        return model.ObjVal, cut
    return None, None


def separation_algorithm(y_bar, theta_bar, S, D, K, dist, N, M, p):
    """
    Algorithm 1: Separation algorithm for Benders cuts.

    Parameters
    ----------
    y_bar : list[float]
        Facility opening solution from master (length M)
    theta_bar : list[float]
        Theta values from master (length N)
    S : dict[int, list[int]]
        S[i] = facilities sorted by distance to client i (closest first)
    D : dict[int, dict[int, float]]
        D[i][k] = k-th smallest distance for client i (1-indexed k)
    K : dict[int, int]
        K[i] = number of distance levels for client i
    dist : array-like
        distance_matrix[i][j] = distance from client i to site j
    N : int
        Number of clients
    M : int
        Number of sites
    p : int
        Number of facilities to open

    Returns
    -------
    cuts_to_add : list[tuple]
        List of (i, cut_expr_data) for violated cuts. cut_expr_data has:
        - 'rhs': float (constant part)
        - 'coeffs': dict j -> coefficient of y_j
    UB : float
        Upper bound = sum over i of OPT(SP_i(y_bar))
    """
    UB = 0.0
    cuts_to_add = []
    tol = 1e-6

    for i in range(N):
        k_i = compute_k_i(i, y_bar, S[i], dist, p, M)

        # Equation 18: OPT(SP(y_bar)) for client i.
        # k_i = number of distance levels passed; allocation cost = D^{k_i+1}
        k_level = min(k_i + 1, K[i])
        opt_sp = D[i][k_level]

        UB += opt_sp

        # Check violation: theta_i < OPT(SP) - tol
        if theta_bar[i] < opt_sp - tol:
            cut_data = _build_cut_from_k_i(i, k_i, D, K, dist, S)
            cuts_to_add.append((i, cut_data))

    return cuts_to_add, UB


def _build_cut_from_k_i(i, k_i, D, K, dist, S):
    """
    Build Benders cut (20) from k_i using equation (19) for dual solution.

    Eq (19): v̂^k = 0 if k <= k̂_i, else v̂^k = D^k - D^{k-1}
    Cut form (16): theta_i >= D^1_i + nu1*(1 - sum_{j in D1} y_j) - sum_{k>=2} nu_k * sum_{j in Dk} y_j
    """
    # Get facilities at each distance level
    facilities_at_D = {}
    for k in range(1, K[i] + 1):
        Dk = D[i][k]
        facilities_at_D[k] = [
            j for j in S[i]
            if abs(dist[i][j] - Dk) < 1e-8
        ]

    # Dual solution from eq (19)
    nu = {}
    if k_i == 0:
        nu[1] = D[i][2] - D[i][1] if K[i] >= 2 else 0.0
    else:
        nu[1] = 0.0
    for k in range(2, K[i] + 1):
        nu[k] = D[i][k] - D[i][k - 1] if k > k_i else 0.0

    return {
        "D1_i": D[i][1],
        "nu": nu,
        "facilities_at_D": facilities_at_D,
        "k_i": k_i,
    }
