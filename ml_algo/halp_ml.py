"""
Machine Learning Guided Exact Algorithm for Hub Arc Location Problem (HAL4).

Source: HALP_ML.pdf
"""

import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def is_connected(arcs, n):
    """Check whether the hub arc set is connected (HAL4 feasibility)."""
    if len(arcs) == 0:
        return False

    adj = {i: set() for i in range(n)}
    for i, j in arcs:
        adj[i].add(j)
        adj[j].add(i)

    visited = set()
    stack = [arcs[0][0]]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(adj[node] - visited)

    nodes_in_arcs = set()
    for i, j in arcs:
        nodes_in_arcs.add(i)
        nodes_in_arcs.add(j)

    return visited >= nodes_in_arcs


def compute_cost(A, distance, flow, alpha):
    """Compute total transportation cost for hub arc configuration A."""
    n = distance.shape[0]
    hubs = set()
    for i, j in A:
        hubs.add(i)
        hubs.add(j)
    hubs = list(hubs)

    total_cost = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            demand = flow[i][j]
            best = float("inf")
            for k in hubs:
                for l in hubs:
                    cost = distance[i][k] + alpha * distance[k][l] + distance[l][j]
                    if cost < best:
                        best = cost
            total_cost += demand * best

    return total_cost


def extract_features(arcs, distance, flow):
    """Extract per-arc static features."""
    features = []
    for i, j in arcs:
        d = distance[i][j]
        flow_ij = flow[i][j] + flow[j][i]
        degree_i = np.sum(flow[i])
        degree_j = np.sum(flow[j])
        features.append([d, flow_ij, degree_i, degree_j])
    return np.array(features)


def build_set_features(A, all_arcs, arc_features):
    """Build binary feature vector indicating selected arcs."""
    vec = np.zeros(len(all_arcs))
    for idx, arc in enumerate(all_arcs):
        if arc in A:
            vec[idx] = 1
    return vec


def solve_HALP(distance, flow, q, alpha, ml_start=5):
    """
    ML-guided exact HAL4 solver.

    Enumerates all connected hub-arc sets of size q. After ml_start iterations,
    logistic regression reorders remaining candidates by predicted quality.
    """
    n = distance.shape[0]

    all_arcs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_sets = list(combinations(all_arcs, q))
    feasible_sets = [A for A in all_sets if is_connected(A, n)]
    print("Total feasible sets:", len(feasible_sets))

    X = []
    y = []
    model = None
    best_cost = float("inf")
    best_A = None

    for t, A in enumerate(feasible_sets):
        if model is not None:
            scores = []
            for A_temp in feasible_sets:
                x_temp = build_set_features(A_temp, all_arcs, None)
                prob = model.predict_proba([x_temp])[0][1]
                scores.append(prob)
            order = np.argsort(scores)[::-1]
            feasible_sets = [feasible_sets[i] for i in order]
            A = feasible_sets[t]

        cost = compute_cost(A, distance, flow, alpha)
        if cost < best_cost:
            best_cost = cost
            best_A = A

        x_feat = build_set_features(A, all_arcs, None)
        label = 1 if A == best_A else 0
        X.append(x_feat)
        y.append(label)

        if t >= ml_start and len(set(y)) > 1:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ])
            model.fit(X, y)

        print(f"Iter {t}, Cost={cost:.2f}, Best={best_cost:.2f}")

    return best_A, best_cost


if __name__ == "__main__":
    distance = np.array([
        [0, 2, 6, 7],
        [2, 0, 5, 3],
        [6, 5, 0, 2],
        [7, 3, 2, 0],
    ], dtype=float)

    flow = np.array([
        [0, 5, 2, 4],
        [5, 0, 3, 6],
        [2, 3, 0, 5],
        [4, 6, 5, 0],
    ], dtype=float)

    q = 2
    alpha = 0.7

    best_A, best_cost = solve_HALP(distance, flow, q, alpha)
    print("\nOptimal hub arcs:", best_A)
    print("Optimal cost:", best_cost)
