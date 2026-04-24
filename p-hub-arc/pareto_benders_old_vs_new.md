# Pareto Benders (Old vs New) for p-Hub-Arc

This note compares the two benchmark entry points:

- `pareto_benders_hub_arc_old.py`
- `pareto_benders_hub_arc_new.py`

Both use the same p-Hub-Arc master, the same callback flow, and the same epsilon-perturbation separation logic.  
The only difference is how the **Magnanti-Wong core point** is chosen.

---

## 1) Common separation form (both versions)

For each OD pair `(i,j)`, dual separation solves:

\[
\max_{\nu \in F_{ij}} f_{ij}(\nu, \tilde y)
\]

with shifted point:

\[
\tilde y = \bar y + \varepsilon y^{core}, \qquad \varepsilon = 10^{-6}
\]

where:

- \(\bar y\): incumbent master solution at callback
- \(y^{core}\): core point (this is where old/new differ)
- \(F_{ij}\): dual-feasible region
- \(f_{ij}\): dual objective expression used to build the cut

After solving at \(\tilde y\), cut violation/RHS is recomputed at \(\bar y\).

---

## 2) Old Pareto version

Old core is fixed uniform:

\[
y^{core}_a = \frac{p}{|H|}, \quad \forall a \in H
\]

So:

\[
\tilde y_a = \bar y_a + \varepsilon \frac{p}{|H|}
\]

Interpretation:

- Symmetric, problem-agnostic interior core.
- No adaptation to instance-specific Phase-1 LP pattern.

Implementation entry:

- `solve_benders_pareto_hub_arc_old(...)`

---

## 3) New Pareto version

Let:

- \(u_a = \frac{p}{|H|}\) (uniform core),
- \(y^{LP}\) = final Phase-1 LP master solution,
- \(\beta \in [0,1]\) = `core_lp_blend`.

New core is:

\[
y^{core}_a = (1-\beta)u_a + \beta y^{LP}_a, \quad \forall a \in H
\]

Then:

\[
\tilde y = \bar y + \varepsilon y^{core}
\]

Interpretation:

- \(\beta=0\): exactly old method.
- \(\beta>0\): shifts MW core toward Phase-1 LP geometry.
- Typical benchmark values: \(\beta \in \{0.2, 0.35, 0.5\}\).

Implementation entry:

- `solve_benders_pareto_hub_arc_new(..., core_lp_blend=0.35)`

---

## 4) What is unchanged mathematically

These are identical in old and new:

- Master model and callback structure.
- Dual-feasible set \(F_{ij}\).
- Epsilon perturbation mechanism (\(\varepsilon\)-trick).
- Cut construction from solved dual multipliers.

Only \(y^{core}\) changes.

---

## 5) Benchmark recommendation

Compare:

1. `old` (fixed uniform core)
2. `new, beta=0.2`
3. `new, beta=0.35`
4. `new, beta=0.5`

Track at least:

- wall time
- total callback cuts added
- final objective
- branch-and-bound nodes

Use same seeds and time limits across all runs.

