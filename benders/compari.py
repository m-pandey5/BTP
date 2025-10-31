
import numpy as np
import math
import random
import gurobipy as gp
from gurobipy import GRB, quicksum

# ==========================================================
# Utility: Euclidean distance matrix
# ==========================================================
def generate_distance_matrix(n, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    pts = np.random.rand(n, 2)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = math.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
    return d


# ==========================================================
# Classical MILP formulation for p-median
# ==========================================================
def classical_pmedian(n, p, d):
    m = gp.Model("classical_pmedian")
    m.Params.OutputFlag = 0

    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    y = m.addVars(n, vtype=GRB.BINARY, name="y")

    # Objective
    m.setObjective(quicksum(d[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

    # Constraints
    for i in range(n):
        m.addConstr(quicksum(x[i, j] for j in range(n)) == 1)
    for i in range(n):
        for j in range(n):
            m.addConstr(x[i, j] <= y[j])
    m.addConstr(quicksum(y[j] for j in range(n)) == p)

    m.optimize()
    obj = m.objVal
    y_sol = np.array([y[j].X for j in range(n)])
    return obj, y_sol


# ==========================================================
# Benders decomposition version (simplified, reusing your logic)
# ==========================================================
def benders_pmedian(n, p, d, time_limit=60):
    # Build distinct distance levels
    Di = []
    level_sites = []
    base_dist = np.zeros(n)
    for i in range(n):
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

    EPS = 1e-7

    def separate_cut_for_client(i, y_vals):
        # Convert y_vals to numpy array if needed
        if not isinstance(y_vals, (list, np.ndarray)):
            try:
                y_vals = np.array([y_vals[j] for j in range(len(y_vals))], dtype=float)
            except Exception:
                y_vals = np.array(list(y_vals.values()), dtype=float)

        Di_i = Di[i]
        buckets = level_sites[i]
        mass = 0.0
        ktilde = -1

        for k, sites_k in enumerate(buckets):
            mass += sum(y_vals[j] for j in sites_k)
            if mass < 1.0 - EPS:
                ktilde = k
            else:
                break

        if ktilde < 0:
            rhs_const = Di_i[0]
            coeffs = np.zeros(n)
            return rhs_const, coeffs, ktilde, rhs_const

        D_next = Di_i[ktilde + 1] if ktilde + 1 < len(Di_i) else Di_i[-1]
        coeffs = np.zeros(n)
        thresh = Di_i[ktilde]
        for j in range(n):
            if d[i, j] <= thresh + 1e-12:
                coeffs[j] = (D_next - d[i, j])
        rhs_const = D_next
        rhs_y = rhs_const - float(np.dot(coeffs, y_vals))
        return rhs_const, coeffs, ktilde, rhs_y


    # --- Phase 1 (LP relaxation) ---
    mdl1 = gp.Model("phase1")
    mdl1.Params.OutputFlag = 0
    mdl1.Params.TimeLimit = time_limit
    y = mdl1.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    theta = mdl1.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS)
    mdl1.addConstr(quicksum(y[j] for j in range(n)) == p)
    for i in range(n):
        mdl1.addConstr(theta[i] >= base_dist[i])
    mdl1.setObjective(quicksum(theta[i] for i in range(n)), GRB.MINIMIZE)

    for _ in range(200):
        mdl1.optimize()
        y_sol = np.array([y[j].X for j in range(n)])
        theta_sol = np.array([theta[i].X for i in range(n)])
        cuts_added = 0
        for i in range(n):
            rhs_const, coeffs, ktilde, rhs_y = separate_cut_for_client(i, y_sol)
            if theta_sol[i] + 1e-9 < rhs_y:
                mdl1.addConstr(theta[i] >= rhs_const - quicksum(coeffs[j] * y[j] for j in range(n)))
                cuts_added += 1
        if cuts_added == 0:
            break
    LB = mdl1.ObjVal
    y_hint = np.array([y[j].X for j in range(n)])

    # --- Phase 2 (Integer Master Problem with Lazy Cuts) ---
    def benders_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            y_vals = model.cbGetSolution(model._y)
            theta_vals = model.cbGetSolution(model._theta)
            for i in range(model._n):
                rhs_const, coeffs, ktilde, rhs_y = separate_cut_for_client(i, y_vals)
                if theta_vals[i] + 1e-9 < rhs_y:
                    expr = model._theta[i]
                    for j in range(model._n):
                        if abs(coeffs[j]) > 1e-12:
                            expr -= coeffs[j] * model._y[j]
                    model.cbLazy(expr >= rhs_const)

    mdl2 = gp.Model("phase2")
    mdl2.Params.OutputFlag = 0
    mdl2.Params.LazyConstraints = 1
    mdl2.Params.TimeLimit = time_limit
    y = mdl2.addVars(n, vtype=GRB.BINARY)
    theta = mdl2.addVars(n, lb=0.0)
    mdl2.addConstr(quicksum(y[j] for j in range(n)) == p)
    for i in range(n):
        mdl2.addConstr(theta[i] >= base_dist[i])
    mdl2.setObjective(quicksum(theta[i] for i in range(n)), GRB.MINIMIZE)
    mdl2._y = y
    mdl2._theta = theta
    mdl2._n = n
    mdl2.optimize(benders_callback)
    return mdl2.ObjVal, np.array([y[j].X for j in range(n)])


# ==========================================================
# Comparison runner
# ==========================================================
if __name__ == "__main__":
    n = 10
    p = 3
    d = generate_distance_matrix(n, seed=42)

    obj_classical, y_classical = classical_pmedian(n, p, d)
    obj_benders, y_benders = benders_pmedian(n, p, d)

    print("Classical formulation objective:", obj_classical)
    print("Benders decomposition objective:", obj_benders)
    print("Absolute difference:", abs(obj_classical - obj_benders))

    print("\nFacilities chosen (classical):", np.where(y_classical > 0.5)[0].tolist())
    print("Facilities chosen (Benders):   ", np.where(y_benders > 0.5)[0].tolist())
