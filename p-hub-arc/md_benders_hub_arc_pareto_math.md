# McDaniel-Devine Benders with Pareto Cuts: Human-Readable Math

This explains `md_benders_hub_arc_pareto.py` without LaTeX rendering.

## 1) Core Objects

Same base model objects as standard MD:

- `N = {1..n}`, `H = {(u,v): u != v}`, `p`.
- `W[i,j]` flows and `D[a,b]` distances.
- Arc routing cost:
  `cost_ij(u,v) = W[i,j] * (D[i,u] + D[u,v] + D[v,j])`.
- Distinct sorted cost levels: `C_ij^1 ... C_ij^Kij`.
- Level sets: `L_ij^k = {arcs with cost C_ij^k}`.
- `Omega = {(i,j): i != j and Kij > 1}`.

## 2) Master Model

Variables:

- `y_uv`:
  - Phase 1: continuous in `[0,1]`
  - Phase 2: binary
- `theta_ij >= 0` for `(i,j) in Omega`

Objective:

`minimize sum_{(i,j) in Omega} theta_ij`

Cardinality:

`sum_{(u,v) in H} y_uv = p`

Plus Benders cuts.

## 3) What Changes Here: Pareto Separation

This file uses `separation_pareto(...)` instead of standard separation.

Cut shape is still:

`theta_ij >= C_ij^1`
`          + nu_1 * (1 - sum_{a in L_ij^1} y_a)`
`          - sum_{k=2..Kij} nu_k * sum_{a in L_ij^k} y_a`

But the `nu` coefficients are chosen to make cuts Pareto-stronger (Magnanti-Wong idea).

Supported modes:

- `pareto_method = "two_step"` (exact two-step MW style)
- `pareto_method = "epsilon"` (single-pass epsilon perturbation style)

## 4) Role of Core Point (`y_core`)

Core point used in separation:

`y_core[a] = p / |H|` for all arcs `a in H`.

Purpose:

- reduce weak/duplicate cuts,
- steer separation toward more informative cuts.

## 5) Phase 1 (LP + Pareto Cut Accumulation)

Loop:

1. Solve LP master.
2. Run Pareto separation at current `(y_bar, theta_bar)`.
3. Add violated cuts to LP.
4. Save each cut in `accumulated_cuts`.
5. If `y_bar` fractional, round (top-`p` arcs) and update best `UB`.

Tracks:

- `LB` = best LP objective so far.
- `UB` = best rounded feasible objective so far.

With total time limit, Phase 1 budget is `phase1_fraction * time_limit`.

## 6) Phase 2 (MIP + Preloaded Pareto Cuts + Lazy Pareto Callback)

Phase 2:

- binary `y`,
- preload all `accumulated_cuts` as regular constraints,
- warm-start from Phase 1 incumbent,
- add more Pareto cuts lazily at MIP incumbents (`_MDParetoCutCallback`).

## 7) Final Objective and Return Fields

When optimal:

`Obj = (sum_{(i,j) in Omega} theta_ij*) + (sum_{(i,j): Kij=1} C_ij^1)`

Returns:

- `objective`, `selected_arcs`, `time`, `status`
- `phase1_cuts`
- `pareto_method`

If not optimal: current implementation returns `objective = None`, status `"FAILED"`.

## 8) Difference from the Other Two Files

- vs `md_benders_hub_arc.py`: same two-phase MD structure, but this file uses Pareto-optimal separation.
- vs `new_model_hub_arc_pareto_phase12.py`: both use Pareto in both phases, but this file explicitly follows MD cut accumulation + preloading behavior.
