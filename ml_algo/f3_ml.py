"""
ML-Guided Exact Algorithm for the p-Hub-Arc Problem under the F3 cost
formulation.

This mirrors halp_ml.py but uses F3's cost function:

  * arcs are DIRECTED: (u, v) with u != v
  * cost of OD pair (i, j) when arc set A is selected
        = min over (u, v) in A of  W[i][j] * (D[i][u] + D[u][v] + D[v][j])
  * NO alpha discount on the inter-hub leg (alpha = 1)
  * NO connectivity requirement on A (just |A| = q)

The optimization scheme is identical to halp_ml:
  1. enumerate every directed q-arc set,
  2. evaluate F3 cost,
  3. once enough labelled examples are collected (after `ml_start`
     evaluations) train a logistic regression on the binary arc-indicator
     vectors and reorder the remaining candidates by predicted "is the
     best so far" probability.

Optimality is preserved because every candidate is still examined; ML
only changes the order in which they are scanned, so a better
incumbent is usually discovered earlier.
"""

import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ----------------------------------------------------------------------
# F3 cost evaluation
# ----------------------------------------------------------------------
def f3_cost(A, W, D):
    """
    Total F3 cost when the chosen directed-arc set is A.

    For each OD pair (i, j) the route must traverse one of the selected
    arcs (u, v) as its inter-hub leg.  We pick the cheapest such arc.
    """
    n = W.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            best = float("inf")
            for (u, v) in A:
                c = W[i][j] * (D[i][u] + D[u][v] + D[v][j])
                if c < best:
                    best = c
            total += best
    return total


# ----------------------------------------------------------------------
# Feature vector for the ML re-ranker — one bit per directed arc
# ----------------------------------------------------------------------
def build_set_features(A, all_arcs):
    vec = np.zeros(len(all_arcs))
    A_set = set(A)
    for idx, arc in enumerate(all_arcs):
        if arc in A_set:
            vec[idx] = 1.0
    return vec


# ----------------------------------------------------------------------
# Main solver
# ----------------------------------------------------------------------
def solve_F3_ML(W, D, q, ml_start=5, verbose=True):
    """
    ML-guided exact F3 hub-arc solver.

    Parameters
    ----------
    W : (n, n) array      flows / demands
    D : (n, n) array      distances
    q : int               number of directed arcs to open
    ml_start : int        iteration at which logistic regression kicks in
    verbose : bool        print per-iteration progress

    Returns
    -------
    best_A : list[(u, v)] optimal directed-arc set
    best_cost : float     F3 objective value
    """
    W = np.asarray(W, dtype=float)
    D = np.asarray(D, dtype=float)
    n = W.shape[0]

    all_arcs = [(u, v) for u in range(n) for v in range(n) if u != v]
    all_sets = [list(s) for s in combinations(all_arcs, q)]
    if verbose:
        print(f"Total directed q-arc sets: {len(all_sets)}")

    X, y = [], []
    model = None
    best_cost = float("inf")
    best_A = None

    pending = list(range(len(all_sets)))
    iteration = 0

    while pending:
        # Re-rank remaining candidates by ML score (post-warm-up).
        if model is not None and iteration >= ml_start:
            feats = np.vstack([
                build_set_features(all_sets[idx], all_arcs) for idx in pending
            ])
            probs = model.predict_proba(feats)[:, 1]
            pending = [pending[i] for i in np.argsort(-probs)]

        idx = pending.pop(0)
        A = all_sets[idx]
        cost = f3_cost(A, W, D)

        improved = cost < best_cost
        if improved:
            best_cost = cost
            best_A = A

        # Label = "did this set tie the current best?"  Mirrors halp_ml.
        feat = build_set_features(A, all_arcs)
        X.append(feat)
        y.append(1 if A == best_A else 0)

        iteration += 1

        if iteration >= ml_start and len(set(y)) > 1:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])
            model.fit(np.asarray(X), np.asarray(y))

        if verbose and (iteration % 50 == 0 or not pending or improved):
            print(f"Iter {iteration:4d}  cost={cost:10.3f}  best={best_cost:10.3f}"
                  f"   A={A}")

    return best_A, best_cost


# ----------------------------------------------------------------------
# Demo on the same toy instance used by test_hub_arc_benders_f3.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    W = np.array([
        [0, 2, 3],
        [2, 0, 4],
        [3, 4, 0],
    ], dtype=float)
    D = np.array([
        [0, 5, 2],
        [5, 0, 3],
        [2, 3, 0],
    ], dtype=float)

    best_A, best_cost = solve_F3_ML(W, D, q=2)
    print("\nOptimal directed arcs:", best_A)
    print("Optimal F3 cost      :", best_cost)
