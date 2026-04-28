# New-Model Pareto (Phase 1 + Phase 2): Human-Readable Math

This explains `new_model_hub_arc_pareto_phase12.py` in plain readable notation.

## 1) Shared Model Ingredients

- Nodes: `N = {1..n}`
- Candidate directed arcs: `H = {(u,v): u != v}`
- Open exactly `p` arcs.
- Flow matrix `W[i,j]`, distance matrix `D[a,b]`.

Per OD pair `(i,j)` and arc `(u,v)`:

`cost_ij(u,v) = W[i,j] * (D[i,u] + D[u,v] + D[v,j])`

Preprocessing creates:

- cost levels `C_ij^1 ... C_ij^Kij`
- level sets `L_ij^k`
- active OD set `Omega = {(i,j): i != j and Kij > 1}`

## 2) Master Model

Variables:

- `y_uv`:
  - Phase 1: `0 <= y_uv <= 1`
  - Phase 2: `y_uv in {0,1}`
- `theta_ij >= 0` for `(i,j) in Omega`

Objective:

`minimize sum_{(i,j) in Omega} theta_ij`

Constraint:

`sum_{(u,v) in H} y_uv = p`

## 3) Pareto MW Algorithm (Human-Readable Equations)

For one OD pair `(i,j)`, at current master point `y_bar`, define:

- `K = Kij`
- `L^k = L_ij^k`
- `C^k = C_ij^k`

### 3.1 Dual-feasible system

Dual variables are `nu_k >= 0` for `k = 1..K`, with constraints:

`nu_k - nu_{k+1} <= C^{k+1} - C^k`  for `k = 1..K-1`

### 3.2 Dual objective at current master point (`y_bar`)

`DSP(y_bar, nu) = C^1`
`               + nu_1 * (1 - sum_{a in L^1} y_bar[a])`
`               - sum_{k=2..K} nu_k * sum_{a in L^k} y_bar[a]`

Standard Benders separation chooses any dual-optimal `nu` for this objective.

### 3.3 Benders cut produced by any valid dual solution `nu`

`theta_ij >= C^1`
`          + nu_1 * (1 - sum_{a in L^1} y[a])`
`          - sum_{k=2..K} nu_k * sum_{a in L^k} y[a]`

### 3.4 Core point used by MW selection

`y_core[a] = p / |H|` for all arcs `a in H`

This is an interior fractional point of the master polytope.

### 3.5 Magnanti-Wong (Pareto) selection equation

Let `D*(y_bar)` be the set of dual-optimal solutions for the current `y_bar`.
MW chooses `nu` by:

`maximize   CUT_at_core(nu)`
`where      CUT_at_core(nu) = C^1`
`                              + nu_1 * (1 - sum_{a in L^1} y_core[a])`
`                              - sum_{k=2..K} nu_k * sum_{a in L^k} y_core[a]`
`subject to nu in D*(y_bar)`

Interpretation: among all dual-optimal cuts valid at `y_bar`, pick the one that is strongest at `y_core`.

### 3.6 Two Pareto modes in this code

- `pareto_method = "two_step"`:
  1) solve dual at `y_bar` to get optimal value `z*`;  
  2) maximize `CUT_at_core(nu)` subject to dual-feasible constraints and `DSP(y_bar, nu) = z*`.

- `pareto_method = "epsilon"`:
  use a perturbed objective that combines being optimal at `y_bar` and strong at `y_core` in one pass.

Both modes return a `nu` used in the same cut formula in 3.3.

## 4) Phase 1 (LP + Pareto Loop)

Function: `phase1_solve_mp_pareto(...)`

Loop logic:

1. Solve LP master.
2. Separate violated Pareto cuts at `(y_bar, theta_bar)`.
3. Add cuts and continue until no violated cut.
4. If `y_bar` is fractional, round to top-`p` arcs and compute heuristic UB.

Phase 1 outputs:

- `LB1`: LP lower bound
- `y1`: warm-start integer solution from heuristic/rounding
- `UB1`: heuristic upper bound

## 5) Phase 2 (MIP + Lazy Pareto Callback)

Function: `solve_benders_hub_arc_pareto_phase12(...)`

Phase 2 solves binary master and:

- warm-starts from `y1` when available,
- enables lazy constraints,
- uses `ParetoPhase2Callback` at each `MIPSOL`,
- callback adds violated Pareto cuts lazily.

So this method applies Pareto separation in both phases.

## 6) Time Handling and Solver Parameters

If total `time_limit` is set, remaining time for Phase 2 is:

`T_phase2 = max(0, time_limit - elapsed_time_so_far)`

Important settings:

- `LazyConstraints = 1`
- `MIPGap = 1e-9`

## 7) Final Objective Report

When optimal:

`Obj = (sum_{(i,j) in Omega} theta_ij*) + (sum_{(i,j): Kij=1} C_ij^1)`

Returned fields:

- `objective`, `selected_arcs`, `time`, `status`

If non-optimal, current code returns `objective=None`, `selected_arcs=None`, status `"FAILED"`.

## 8) Difference from the Other Two Files

- vs `md_benders_hub_arc.py`: this method uses Pareto cuts in both phases; standard MD uses non-Pareto cuts.
- vs `md_benders_hub_arc_pareto.py`: both are Pareto-based, but that file follows explicit MD cut accumulation/preload architecture, while this one uses its own phase-1 loop + phase-2 lazy callback design.
