# ======================================================
# Benders Decomposition for p-Median Problem (Gurobi)
# Works in VS Code or terminal with gurobipy installed
# ======================================================

from gurobipy import Model, GRB, quicksum
import math
import random
import numpy as np

# ------------------------------------------------------
# Small utilities
# ------------------------------------------------------
EPS = 1e-7

def argpartition_top_p(values, p):
    """Return indices of the top-p values (largest)."""
    if p <= 0:
        return []
    idx = np.argpartition(values, -p)[-p:]
    return idx[np.argsort(values[idx])[::-1]]


# ------------------------------------------------------
# Demo instance (you can replace with your dataset)
# ------------------------------------------------------
random.seed(0)
np.random.seed(0)

N = 20     # number of clients
M = 20     # number of candidate sites
p = 4      # open exactly p sites

clients_xy = np.random.rand(N, 2)
sites_xy   = np.random.rand(M, 2)

# Euclidean distances
d = np.zeros((N, M))
for i in range(N):
    for j in range(M):
        dx, dy = clients_xy[i,0] - sites_xy[j,0], clients_xy[i,1] - sites_xy[j,1]
        d[i, j] = math.hypot(dx, dy)

assert p <= M, "p cannot exceed M"

# ------------------------------------------------------
# Build distinct distance levels per client
# ------------------------------------------------------
Di = []
level_sites = []
base_dist = np.zeros(N)

for i in range(N):
    dist_i = d[i, :]
    uniq, inv = np.unique(np.round(dist_i, 12), return_inverse=True)
    uniq = uniq.tolist()
    K_i = len(uniq)
    buckets = [[] for _ in range(K_i)]
    for j, k in enumerate(inv):
        buckets[k].append(j)
    Di.append(uniq)
    level_sites.append(buckets)
    base_dist[i] = uniq[0]


# ------------------------------------------------------
# Cut separation helpers
# ------------------------------------------------------
def separate_cut_for_client(i, y_vals):
    Di_i = Di[i]
    buckets = level_sites[i]
    mass = 0.0
    ktilde = -1

    # Convert to plain array if y_vals is a Gurobi tupledict
    if not isinstance(y_vals, (list, np.ndarray)):
        try:
            y_vals = np.array([y_vals[j] for j in range(M)], dtype=float)
        except Exception:
            y_vals = np.array(list(y_vals.values()), dtype=float)

    for k, sites_k in enumerate(buckets):
        mass += sum(y_vals[j] for j in sites_k)
        if mass < 1.0 - EPS:
            ktilde = k
        else:
            break

    if ktilde < 0:
        rhs_const = Di_i[0]
        coeffs = np.zeros(M)
        return rhs_const, coeffs, ktilde, rhs_const

    D_next = Di_i[ktilde + 1] if ktilde + 1 < len(Di_i) else Di_i[-1]
    coeffs = np.zeros(M)
    thresh = Di_i[ktilde]
    for j in range(M):
        if d[i, j] <= thresh + 1e-12:
            coeffs[j] = (D_next - d[i, j])

    rhs_const = D_next
    rhs_y = rhs_const - float(np.dot(coeffs, y_vals))
    return rhs_const, coeffs, ktilde, rhs_y



def add_cut_to_model(model, theta_vars, y_vars, i, rhs_const, coeffs):
    """Add cut: theta_i >= rhs_const - sum_j coeffs[j]*y_j"""
    model.addConstr(
        theta_vars[i] >= rhs_const - quicksum(coeffs[j]*y_vars[j] for j in range(len(y_vars))),
        name=f"BD_i_{i}_{model.getAttr('NumConstrs')}"
    )


# ------------------------------------------------------
# Phase-1: LP relaxation
# ------------------------------------------------------
def phase1_lp_benders(N, M, p, d, Di, level_sites, time_limit_sec=60, max_iters=200):
    model = Model("MP_phase1")
    model.Params.OutputFlag = 1
    if time_limit_sec is not None:
        model.Params.TimeLimit = time_limit_sec

    y = model.addVars(M, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
    theta = model.addVars(N, lb=0.0, vtype=GRB.CONTINUOUS, name="theta")

    model.addConstr(quicksum(y[j] for j in range(M)) == p, name="open_p")

    for i in range(N):
        model.addConstr(theta[i] >= base_dist[i], name=f"basecut_{i}")

    model.setObjective(quicksum(theta[i] for i in range(N)), GRB.MINIMIZE)

    UB = float("inf")
    y_incumbent = None

    for it in range(1, max_iters + 1):
        model.optimize()
        if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            break

        y_sol = np.array([y[j].X for j in range(M)])
        theta_sol = np.array([theta[i].X for i in range(N)])
        cuts_added = 0

        for i in range(N):
            rhs_const, coeffs, ktilde, rhs_y = separate_cut_for_client(i, y_sol)
            if theta_sol[i] + 1e-9 < rhs_y:
                add_cut_to_model(model, theta, y, i, rhs_const, coeffs)
                cuts_added += 1

        idx = argpartition_top_p(y_sol, p)
        y_round = np.zeros(M)
        y_round[idx] = 1.0

        assign_cost = 0.0
        open_idx = np.where(y_round > 0.5)[0]
        for i in range(N):
            assign_cost += np.min(d[i, open_idx])
        if assign_cost < UB - 1e-9:
            UB = assign_cost
            y_incumbent = y_round.copy()

        if cuts_added == 0:
            break

    LB = model.ObjVal if model.SolCount > 0 else float("inf")
    return LB, UB, y_incumbent, model


# ------------------------------------------------------
# Phase-2: Integer master + lazy cuts
# ------------------------------------------------------
def build_mp_integer(N, M, p):
    mdl = Model("MP_phase2")
    mdl.Params.OutputFlag = 1
    mdl.Params.LazyConstraints = 1
    y = mdl.addVars(M, vtype=GRB.BINARY, name="y")
    theta = mdl.addVars(N, lb=0.0, vtype=GRB.CONTINUOUS, name="theta")
    mdl.addConstr(quicksum(y[j] for j in range(M)) == p, name="open_p")
    for i in range(N):
        mdl.addConstr(theta[i] >= base_dist[i], name=f"basecut_{i}")
    mdl.setObjective(quicksum(theta[i] for i in range(N)), GRB.MINIMIZE)
    return mdl, y, theta


def benders_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        y_vals = model.cbGetSolution(model._y)
        theta_vals = model.cbGetSolution(model._theta)
        for i in range(model._N):
            rhs_const, coeffs, ktilde, rhs_y = separate_cut_for_client(i, y_vals)
            if theta_vals[i] + 1e-9 < rhs_y:
                expr = model._theta[i]
                for j in range(model._M):
                    if abs(coeffs[j]) > 0:
                        expr -= coeffs[j] * model._y[j]
                model.cbLazy(expr >= rhs_const)


def solve_phase2(N, M, p, d, Di, level_sites, UB_hint=None, y_hint=None, time_limit_sec=300):
    mdl, y, theta = build_mp_integer(N, M, p)
    if time_limit_sec is not None:
        mdl.Params.TimeLimit = time_limit_sec

    if UB_hint is not None and y_hint is not None:
        open_idx = np.where(y_hint > 0.5)[0]
        theta_hint = np.array([np.min(d[i, open_idx]) for i in range(N)])
        for j in range(M):
            y[j].Start = float(y_hint[j])
        for i in range(N):
            theta[i].Start = float(theta_hint[i])
        mdl.update()

    mdl._N = N
    mdl._M = M
    mdl._y = y
    mdl._theta = theta

    mdl.optimize(benders_callback)
    return mdl, y, theta


# ------------------------------------------------------
# Run the two-phase algorithm
# ------------------------------------------------------
if __name__ == "__main__":
    LB1, UB1, y1, mp_lp_model = phase1_lp_benders(N, M, p, d, Di, level_sites, time_limit_sec=60, max_iters=200)
    print(f"[Phase-1] LB1 (LP): {LB1:.6f}, UB1 (rounded): {UB1:.6f}")

    mdl2, y_vars, theta_vars = solve_phase2(N, M, p, d, Di, level_sites, UB_hint=UB1, y_hint=y1, time_limit_sec=300)
    if mdl2.SolCount > 0:
        print(f"[Phase-2] Obj: {mdl2.ObjVal:.6f}, Gap: {mdl2.MIPGap:.6%}")

        y_opt = np.array([y_vars[j].X for j in range(M)])
        open_idx = np.where(y_opt > 0.5)[0]
        print("Open sites:", open_idx.tolist())

        assign_cost = 0.0
        for i in range(N):
            assign_cost += np.min(d[i, open_idx])
        print(f"Recomputed objective: {assign_cost:.6f}")
    else:
        print(f"[Phase-2] No solution (status {mdl2.Status})")
