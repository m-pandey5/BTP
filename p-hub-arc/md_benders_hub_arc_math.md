# McDaniel-Devine Benders (Standard Cuts): Human-Readable Math

This explains `md_benders_hub_arc.py` in plain notation (no LaTeX rendering needed).

## 1) Sets, Data, and Preprocessing

- `N = {1..n}`: nodes.
- `H = {(u,v): u != v}`: directed candidate hub arcs.
- `p`: number of arcs to open.
- `W[i,j]`: flow from origin `i` to destination `j`.
- `D[a,b]`: distance/cost between nodes.

Per OD pair `(i,j)` and hub arc `(u,v)`:

`cost_ij(u,v) = W[i,j] * ( D[i,u] + D[u,v] + D[v,j] )`

For each OD pair `(i,j)`, preprocessing builds:

- `C_ij^1 < C_ij^2 < ... < C_ij^Kij`: sorted unique cost levels.
- `L_ij^k`: set of arcs with cost exactly `C_ij^k`.
- `Kij`: number of distinct levels.

Define:

- `Omega = {(i,j): i != j and Kij > 1}`.

Pairs with `Kij = 1` have fixed cost `C_ij^1` and do not need theta variables.

## 2) Master Problem (Both Phases)

Variables:

- `y_uv`:
  - Phase 1 LP: `0 <= y_uv <= 1`
  - Phase 2 MIP: `y_uv in {0,1}`
- `theta_ij >= 0` for `(i,j) in Omega`

Master objective:

`minimize   sum_{(i,j) in Omega} theta_ij`

Constraint:

`sum_{(u,v) in H} y_uv = p`

Plus Benders optimality cuts.

## 3) Dual Subproblem and Standard Benders Cut

At a master point `y_bar`, for each `(i,j) in Omega`, solve dual with variables
`nu_1, ..., nu_Kij` where each `nu_k >= 0`.

Dual objective:

`maximize  C_ij^1`
`        + nu_1 * (1 - sum_{a in L_ij^1} y_bar[a])`
`        - sum_{k=2..Kij} nu_k * sum_{a in L_ij^k} y_bar[a]`

Dual constraints (for k = 1..Kij-1):

`nu_k - nu_{k+1} <= C_ij^{k+1} - C_ij^k`

If dual value is `z_ij(y_bar)`, add cut:

`theta_ij >= C_ij^1`
`          + nu_1 * (1 - sum_{a in L_ij^1} y_a)`
`          - sum_{k=2..Kij} nu_k * sum_{a in L_ij^k} y_a`

This is exactly what `_cut_expr(...)` builds.

## 4) Phase 1 (LP Cutting-Plane, McDaniel-Devine Style)

Repeat:

1. Solve LP master.
2. Separate violated cuts for every `(i,j) in Omega`.
3. Add cuts to LP master.
4. Save every generated cut in `accumulated_cuts` (for Phase 2).
5. If `y` is fractional, round by picking top-`p` arcs by `y` value.

Bounds tracked:

- `LB`: best LP bound seen.
- `UB`: best rounded feasible objective seen.

Time split:

- If `time_limit` is provided, Phase 1 uses `phase1_fraction * time_limit`.

## 5) Phase 2 (Branch-and-Benders-Cut MIP)

Phase 2 solves binary master and:

- pre-loads all `accumulated_cuts` as regular constraints,
- warm-starts from Phase 1 rounded solution,
- adds more violated cuts lazily at `MIPSOL` callback points.

This pre-load behavior is the key McDaniel-Devine implementation detail in this file.

## 6) Final Objective Reported

When optimal:

`Obj = (sum_{(i,j) in Omega} theta_ij*) + (sum_{(i,j): Kij=1} C_ij^1)`

Returned fields:

- `objective`
- `selected_arcs`
- `phase1_cuts`
- `time`, `status`

If not optimal, current code returns `objective = None` and status `"FAILED"`.

## 7) Difference from the Other Two Files

- vs `md_benders_hub_arc_pareto.py`: same two-phase MD framework, but this file uses standard (non-Pareto) separation.
- vs `new_model_hub_arc_pareto_phase12.py`: this file explicitly accumulates Phase 1 cuts and pre-loads them in Phase 2.
