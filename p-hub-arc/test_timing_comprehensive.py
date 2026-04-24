"""
Comprehensive Timing Test: F3 vs Benders for p-Hub-Arc

Compares:
  - F3 (direct canonical formulation)
  - Normal Benders (HubArcBenders: master + lazy callback, no Phase 1)
  - New Benders (solve_benders_hub_arc: Phase 1 LP + Phase 2 with warm start)

Run: python test_timing_comprehensive.py
      or: python -m pytest p-hub-arc/test_timing_comprehensive.py -v
"""

import sys
import os
import threading
import numpy as np
import time
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from hub_arc_models import solve_hub_arc_F3, HubArcBenders
from new_model_hub_arc import solve_benders_hub_arc, preprocess
from md_benders_hub_arc import solve_md_benders_hub_arc
from md_benders_hub_arc_pareto import solve_md_benders_hub_arc_pareto
from new_model_hub_arc_pareto_phase12 import solve_benders_hub_arc_pareto_phase12


@contextmanager
def _long_task_heartbeat(label: str, interval_sec: float):
    """Print a line every `interval_sec` so long Gurobi runs do not look hung."""
    if interval_sec <= 0:
        yield
        return
    stop = threading.Event()
    t0 = time.time()

    def _loop():
        n_hb = 0
        while not stop.wait(interval_sec):
            n_hb += 1
            elapsed = int(time.time() - t0)
            print(
                f"\n      [running] {label}  (heartbeat #{n_hb}, ~{elapsed}s elapsed)",
                flush=True,
            )

    th = threading.Thread(target=_loop, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=1.0)


def build_instance(n: int, p: int, seed: Optional[int] = None) -> Tuple[List, List]:
    """Build random W (flows) and D (distances) for hub-arc."""
    if seed is not None:
        np.random.seed(seed)
    W = np.random.rand(n, n) * 10
    np.fill_diagonal(W, 0)
    D = np.random.rand(n, n) * 20
    np.fill_diagonal(D, 0)
    return W.tolist(), D.tolist()


def _all_objectives_close(objectives: List[Optional[float]], tol: float = 1e-4) -> bool:
    """True if all non-None objectives agree within tol (or there is at most one)."""
    finite = [o for o in objectives if o is not None]
    if not finite:
        return False
    if len(finite) == 1:
        return True
    return max(finite) - min(finite) < tol


def run_instance(
    n: int,
    p: int,
    W: List,
    D: List,
    time_limit: Optional[float] = None,
    use_phase1: bool = True,
    stage_progress: bool = False,
    heartbeat_sec: float = 0.0,
    skip_f3: bool = False,
    include_md_and_phase12: bool = False,
) -> Dict[str, Any]:
    """
    Run F3, normal Benders (HubArcBenders), and new Benders on one instance.
    Optionally also run McDaniel-Devine (MD), MD+Pareto, and new-model Pareto phase12.

    If stage_progress is True, prints a short line before each solver (flush=True);
    F3 at medium n can take many minutes with no other output.

    If heartbeat_sec > 0 and stage_progress is True, prints periodic heartbeats
    during long steps (F3 / Norm / New / extra solvers) so the process does not look stuck.

    skip_f3 : if True, do not run the F3 MIP. Use for large n: F3 model build
    in pure Python is huge (minutes+) before Gurobi TimeLimit even applies.
    When skipped, "match" (without extras) compares Normal vs New Benders only.

    include_md_and_phase12 : if True, after New Benders also run, in order,
    solve_md_benders_hub_arc, solve_md_benders_hub_arc_pareto (pareto two_step),
    and solve_benders_hub_arc_pareto_phase12. "match" then requires all
    non-None objectives among the selected solvers to agree within 1e-4
    (F3 included when not skip_f3; otherwise the five Benders methods).
    """
    hb = stage_progress and heartbeat_sec > 0
    if skip_f3:
        f3_res = {
            "objective": None,
            "time": None,
            "status": "SKIPPED",
        }
        if stage_progress:
            print(
                "(F3 skipped) | ",
                end="",
                flush=True,
            )
    else:
        if stage_progress:
            print("(F3 …)", end=" ", flush=True)
        if hb:
            with _long_task_heartbeat("F3 direct MIP (OutputFlag=0, may be slow)", heartbeat_sec):
                f3_res = solve_hub_arc_F3(
                    n, p, W, D, gurobi_output=False, time_limit=time_limit
                )
        else:
            f3_res = solve_hub_arc_F3(
                n, p, W, D, gurobi_output=False, time_limit=time_limit
            )
        if stage_progress:
            f3t_done = f3_res.get("time") or 0.0
            print(f"done {f3t_done:.2f}s | ", end="", flush=True)

    # Normal Benders (HubArcBenders): no Phase 1, callback-only
    if stage_progress:
        print("(Norm Benders …)", end=" ", flush=True)
    b_norm = HubArcBenders(n=n, p=p, W=W, D=D, verbose=False)
    if hb:
        with _long_task_heartbeat("Normal Benders (MIP + callbacks)", heartbeat_sec):
            norm_res = b_norm.solve(time_limit=time_limit)
    else:
        norm_res = b_norm.solve(time_limit=time_limit)
    if stage_progress:
        nt_done = norm_res.get("time") or 0.0
        print(f"done {nt_done:.2f}s | ", end="", flush=True)
    # HubArcBenders returns master.ObjVal; add constant for OD pairs with K==1
    _, C, _, K, _, _, _ = preprocess(n, W, D)
    if norm_res["objective"] is not None:
        add_const = sum(C[(i, j)][0] for (i, j) in K if i != j and K[(i, j)] == 1)
        norm_res = dict(norm_res, objective=norm_res["objective"] + add_const)

    # New Benders (Phase 1 + Phase 2)
    if stage_progress:
        print("(New Benders …)", end=" ", flush=True)
    if hb:
        with _long_task_heartbeat("New Benders (Phase1 LP + Phase2 MIP)", heartbeat_sec):
            new_res = solve_benders_hub_arc(
                n, p, W, D, verbose=False, use_phase1=use_phase1, time_limit=time_limit
            )
    else:
        new_res = solve_benders_hub_arc(
            n, p, W, D, verbose=False, use_phase1=use_phase1, time_limit=time_limit
        )
    if stage_progress:
        nnt_done = new_res.get("time") or 0.0
        if include_md_and_phase12:
            print(f"done {nnt_done:.2f}s | ", end="", flush=True)
        else:
            print(f"done {nnt_done:.2f}s", flush=True)

    md_res: Optional[Dict[str, Any]] = None
    mdp_res: Optional[Dict[str, Any]] = None
    p12_res: Optional[Dict[str, Any]] = None
    if include_md_and_phase12:
        if stage_progress:
            print("(MD Benders …)", end=" ", flush=True)
        if hb:
            with _long_task_heartbeat("MD Benders (McDaniel-Devine, standard cuts)", heartbeat_sec):
                md_res = solve_md_benders_hub_arc(
                    n, p, W, D, time_limit=time_limit, verbose=False, use_phase1=use_phase1
                )
        else:
            md_res = solve_md_benders_hub_arc(
                n, p, W, D, time_limit=time_limit, verbose=False, use_phase1=use_phase1
            )
        if stage_progress:
            print(f"done {md_res.get('time') or 0.0:.2f}s | ", end="", flush=True)

        if stage_progress:
            print("(MD Pareto …)", end=" ", flush=True)
        if hb:
            with _long_task_heartbeat("MD Pareto Benders (two_step)", heartbeat_sec):
                mdp_res = solve_md_benders_hub_arc_pareto(
                    n, p, W, D, time_limit=time_limit, verbose=False, use_phase1=use_phase1
                )
        else:
            mdp_res = solve_md_benders_hub_arc_pareto(
                n, p, W, D, time_limit=time_limit, verbose=False, use_phase1=use_phase1
            )
        if stage_progress:
            print(f"done {mdp_res.get('time') or 0.0:.2f}s | ", end="", flush=True)

        if stage_progress:
            print("(Pareto phase12 …)", end=" ", flush=True)
        if hb:
            with _long_task_heartbeat("New-model Pareto phase12 (two_step)", heartbeat_sec):
                p12_res = solve_benders_hub_arc_pareto_phase12(
                    n, p, W, D, time_limit=time_limit, verbose=False,
                    use_phase1=use_phase1, pareto_method="two_step",
                )
        else:
            p12_res = solve_benders_hub_arc_pareto_phase12(
                n, p, W, D, time_limit=time_limit, verbose=False,
                use_phase1=use_phase1, pareto_method="two_step",
            )
        if stage_progress:
            print(f"done {p12_res.get('time') or 0.0:.2f}s", flush=True)

    ref = f3_res["objective"]
    diff_norm = diff_new = None
    if include_md_and_phase12 and md_res is not None and mdp_res is not None and p12_res is not None:
        to_compare: List[Optional[float]] = []
        if not skip_f3:
            to_compare.append(ref)
        to_compare.extend(
            [
                norm_res["objective"],
                new_res["objective"],
                md_res.get("objective"),
                mdp_res.get("objective"),
                p12_res.get("objective"),
            ]
        )
        match_f3 = _all_objectives_close(to_compare)
    else:
        if skip_f3:
            match_f3 = (
                norm_res["objective"] is not None
                and new_res["objective"] is not None
                and abs(norm_res["objective"] - new_res["objective"]) < 1e-4
            )
        else:
            match_f3 = ref is not None
            if ref is not None:
                if norm_res["objective"] is not None:
                    diff_norm = abs(ref - norm_res["objective"])
                    match_f3 = match_f3 and diff_norm < 1e-4
                if new_res["objective"] is not None:
                    diff_new = abs(ref - new_res["objective"])
                    match_f3 = match_f3 and diff_new < 1e-4

    out: Dict[str, Any] = {
        "n": n,
        "p": p,
        "f3_obj": ref,
        "f3_time": f3_res["time"],
        "f3_status": f3_res["status"],
        "norm_obj": norm_res["objective"],
        "norm_time": norm_res["time"],
        "norm_status": norm_res["status"],
        "new_obj": new_res["objective"],
        "new_time": new_res["time"],
        "new_status": new_res["status"],
        "match": match_f3,
        "diff_norm": diff_norm,
        "diff_new": diff_new,
    }
    if include_md_and_phase12 and md_res is not None and mdp_res is not None and p12_res is not None:
        out["md_time"] = md_res.get("time")
        out["md_obj"] = md_res.get("objective")
        out["md_status"] = md_res.get("status")
        out["mdp_time"] = mdp_res.get("time")
        out["mdp_obj"] = mdp_res.get("objective")
        out["mdp_status"] = mdp_res.get("status")
        out["p12_time"] = p12_res.get("time")
        out["p12_obj"] = p12_res.get("objective")
        out["p12_status"] = p12_res.get("status")
    return out


def run_timing_suite(
    instance_specs: List[Dict],
    seeds: List[int],
    time_limit: Optional[float] = 300.0,
    use_phase1: bool = True,
    verbose: bool = True,
    heartbeat_sec: float = 0.0,
    skip_f3: bool = False,
    include_md_and_phase12: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run full timing suite: each (n, p) × each seed.
    Returns list of results with aggregated stats per (n, p).

    heartbeat_sec : if > 0 and verbose, print periodic heartbeats during each
    solver step (F3 can run a long time with no Gurobi log when OutputFlag=0).
    Set to 0 to disable.

    skip_f3 : if True, skip the F3 direct MIP (see run_instance).

    include_md_and_phase12 : if True, also run MD, MD+Pareto, and Pareto phase12
    after New Benders (see run_instance).
    """
    results = []
    idx = 0

    for spec in instance_specs:
        n = spec["n"]
        p = spec["p"]
        fixed = spec.get("fixed", False)
        seed_list = [None] if fixed else seeds

        for seed in seed_list:
            idx += 1
            if verbose:
                label = f"n={n}, p={p}" + (f", seed={seed}" if seed is not None else " (fixed)")
                print(f"  [{idx}] {label} ...", end=" ", flush=True)

            if fixed:
                W, D = spec["W"], spec["D"]
            else:
                W, D = build_instance(n, p, seed)

            try:
                r = run_instance(
                    n,
                    p,
                    W,
                    D,
                    time_limit=time_limit,
                    use_phase1=use_phase1,
                    stage_progress=verbose,
                    heartbeat_sec=heartbeat_sec if verbose else 0.0,
                    skip_f3=skip_f3,
                    include_md_and_phase12=include_md_and_phase12,
                )
                r["seed"] = seed
                r["fixed"] = fixed
                results.append(r)
                if verbose:
                    f3t, nt, nn = r["f3_time"], r["norm_time"], r["new_time"]
                    times: List[Tuple[str, Any]] = [
                        ("F3", f3t), ("Norm", nt), ("New", nn),
                    ]
                    if include_md_and_phase12:
                        times.extend(
                            [
                                ("MD", r.get("md_time")),
                                ("MDP", r.get("mdp_time")),
                                ("P12", r.get("p12_time")),
                            ]
                        )
                    valid_times = [
                        (name, t) for name, t in times
                        if t is not None and t >= 0
                    ]
                    fastest = min(valid_times, key=lambda x: x[1])[0] if valid_times else "-"
                    f3o, no, neo = r.get("f3_obj"), r.get("norm_obj"), r.get("new_obj")
                    f3o_s = f"{f3o:.4f}" if f3o is not None else "N/A"
                    no_s = f"{no:.4f}" if no is not None else "N/A"
                    neo_s = f"{neo:.4f}" if neo is not None else "N/A"
                    f3s = f"{f3t:.3f}" if f3t is not None else "N/A"
                    match_str = "Match" if r.get("match") else "MISMATCH"
                    if include_md_and_phase12:
                        mdt, mdp_t, p12t = r.get("md_time"), r.get("mdp_time"), r.get("p12_time")
                        mdt_s = f"{mdt:.3f}" if mdt is not None else "N/A"
                        mdp_s = f"{mdp_t:.3f}" if mdp_t is not None else "N/A"
                        p12_s = f"{p12t:.3f}" if p12t is not None else "N/A"
                        print(
                            f"F3={f3s}s, NormB={nt:.3f}s, NewB={nn:.3f}s, "
                            f"MD={mdt_s}s, MDP={mdp_s}s, P12={p12_s}s (fastest: {fastest})"
                        )
                        mdo = r.get("md_obj")
                        mdpo = r.get("mdp_obj")
                        p12o = r.get("p12_obj")
                        mdo_s = f"{mdo:.4f}" if mdo is not None else "N/A"
                        mdpo_s = f"{mdpo:.4f}" if mdpo is not None else "N/A"
                        p12o_s = f"{p12o:.4f}" if p12o is not None else "N/A"
                        print(
                            f"      Obj: F3={f3o_s}  NormB={no_s}  NewB={neo_s}  "
                            f"MD={mdo_s}  MDP={mdpo_s}  P12={p12o_s}  |  {match_str}"
                        )
                    else:
                        print(
                            f"F3={f3s}s, NormB={nt:.3f}s, NewB={nn:.3f}s (fastest: {fastest})"
                        )
                        print(
                            f"      Obj: F3={f3o_s}  NormB={no_s}  NewB={neo_s}  |  {match_str}"
                        )
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                err_row: Dict[str, Any] = {
                    "n": n, "p": p, "seed": seed, "fixed": fixed,
                    "f3_time": None, "norm_time": None, "new_time": None,
                    "match": False, "error": str(e),
                }
                if include_md_and_phase12:
                    err_row["md_time"] = err_row["mdp_time"] = err_row["p12_time"] = None
                results.append(err_row)

    return results


def aggregate_by_instance(results: List[Dict]) -> Dict[Tuple[int, int], Dict]:
    """Aggregate results by (n, p): mean time, std, count, match rate."""
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        key = (r["n"], r["p"])
        by_key[key].append(r)

    agg = {}
    for (n, p), rows in by_key.items():
        f3_t = [x["f3_time"] for x in rows if x.get("f3_time") is not None]
        norm_t = [x["norm_time"] for x in rows if x.get("norm_time") is not None]
        new_t = [x["new_time"] for x in rows if x.get("new_time") is not None]
        md_t = [x["md_time"] for x in rows if x.get("md_time") is not None]
        mdp_t = [x["mdp_time"] for x in rows if x.get("mdp_time") is not None]
        p12_t = [x["p12_time"] for x in rows if x.get("p12_time") is not None]
        matches = sum(1 for x in rows if x.get("match"))
        agg[(n, p)] = {
            "n": n,
            "p": p,
            "count": len(rows),
            "f3_mean": np.mean(f3_t) if f3_t else None,
            "norm_mean": np.mean(norm_t) if norm_t else None,
            "new_mean": np.mean(new_t) if new_t else None,
            "md_mean": np.mean(md_t) if md_t else None,
            "mdp_mean": np.mean(mdp_t) if mdp_t else None,
            "p12_mean": np.mean(p12_t) if p12_t else None,
            "match_count": matches,
            "match_rate": matches / len(rows) if rows else 0,
        }
    return agg


def print_timing_table(results: List[Dict], agg: Dict):
    """Print formatted timing comparison table."""
    has_extras = any(
        "error" not in r and r.get("md_time") is not None for r in results
    ) or any(a.get("md_mean") is not None for a in agg.values())

    print("\n" + "=" * 115)
    print("TIMING COMPARISON TABLE (F3 vs Normal Benders vs New Benders)")
    print("=" * 115)

    header = f"{'n':>4} | {'p':>3} | {'#runs':>6} | {'F3 (s)':>10} | {'Norm Benders (s)':>16} | {'New Benders (s)':>16} | {'Fastest':>12} | Match"
    print(header)
    print("-" * 115)

    for key in sorted(agg.keys(), key=lambda k: (k[0], k[1])):
        a = agg[key]
        n, p = a["n"], a["p"]
        fm = a["f3_mean"]
        nm = a["norm_mean"]
        newm = a["new_mean"]
        count = a["count"]

        f_str = f"{fm:.4f}" if fm is not None else "   N/A   "
        n_str = f"{nm:.4f}" if nm is not None else "   N/A   "
        new_str = f"{newm:.4f}" if newm is not None else "   N/A   "

        times = [("F3", fm), ("NormB", nm), ("NewB", newm)]
        if has_extras:
            times.extend(
                [
                    ("MD", a.get("md_mean")),
                    ("MDP", a.get("mdp_mean")),
                    ("P12", a.get("p12_mean")),
                ]
            )
        valid = [(name, t) for name, t in times if t is not None and t >= 0]
        fastest = min(valid, key=lambda x: x[1])[0] if valid else "-"
        match_str = f"{a['match_count']}/{count}"
        print(f"{n:>4} | {p:>3} | {count:>6} | {f_str:>10} | {n_str:>16} | {new_str:>16} | {fastest:>12} | {match_str}")

    if has_extras:
        print("\n" + "=" * 115)
        print("MEAN TIMES — MD, MD Pareto, Pareto phase12 (only when those solvers were run)")
        print("=" * 115)
        h2 = f"{'n':>4} | {'p':>3} | {'#runs':>6} | {'MD (s)':>10} | {'MD Pareto (s)':>14} | {'P12 (s)':>10} | {'Fastest3':>12}"
        print(h2)
        print("-" * 115)
        for key in sorted(agg.keys(), key=lambda k: (k[0], k[1])):
            a = agg[key]
            n, p = a["n"], a["p"]
            count = a["count"]
            mdm = a.get("md_mean")
            mdp = a.get("mdp_mean")
            p12m = a.get("p12_mean")
            ms = f"{mdm:.4f}" if mdm is not None else "   N/A   "
            mps = f"{mdp:.4f}" if mdp is not None else "   N/A   "
            pss = f"{p12m:.4f}" if p12m is not None else "   N/A   "
            t3 = [("MD", mdm), ("MDP", mdp), ("P12", p12m)]
            v3 = [(n2, t) for n2, t in t3 if t is not None and t >= 0]
            f3 = min(v3, key=lambda x: x[1])[0] if v3 else "-"
            print(f"{n:>4} | {p:>3} | {count:>6} | {ms:>10} | {mps:>14} | {pss:>10} | {f3:>12}")


def print_detailed_results(results: List[Dict]):
    """Print per-run details (times, objectives, match)."""
    print("\n" + "=" * 115)
    print("DETAILED PER-RUN RESULTS (times, objectives, match)")
    print("=" * 115)
    for i, r in enumerate(results):
        if "error" in r:
            print(f"  [{i+1}] n={r['n']}, p={r['p']}: ERROR - {r['error']}")
            continue
        status = "Match" if r.get("match") else "MISMATCH"
        f3t, nt, nn = r.get("f3_time"), r.get("norm_time"), r.get("new_time")
        f3o, no, neo = r.get("f3_obj"), r.get("norm_obj"), r.get("new_obj")
        f3s = f"{f3t:.4f}" if f3t is not None else "N/A"
        ns = f"{nt:.4f}" if nt is not None else "N/A"
        nns = f"{nn:.4f}" if nn is not None else "N/A"
        f3o_s = f"{f3o:.4f}" if f3o is not None else "N/A"
        no_s = f"{no:.4f}" if no is not None else "N/A"
        neo_s = f"{neo:.4f}" if neo is not None else "N/A"
        print(f"  [{i+1}] n={r['n']}, p={r['p']}, seed={r.get('seed')}: "
              f"F3={f3s}s, NormB={ns}s, NewB={nns}s")
        print(f"       Obj: F3={f3o_s}  NormB={no_s}  NewB={neo_s}  |  {status}")
        if "md_time" in r and "error" not in r:
            mdt, mdp_t, p12t = r.get("md_time"), r.get("mdp_time"), r.get("p12_time")
            mds = f"{mdt:.4f}" if mdt is not None else "N/A"
            mdps = f"{mdp_t:.4f}" if mdp_t is not None else "N/A"
            p12s = f"{p12t:.4f}" if p12t is not None else "N/A"
            mdo, mdpo, p12o = r.get("md_obj"), r.get("mdp_obj"), r.get("p12_obj")
            mdos = f"{mdo:.4f}" if mdo is not None else "N/A"
            mdpos = f"{mdpo:.4f}" if mdpo is not None else "N/A"
            p12os = f"{p12o:.4f}" if p12o is not None else "N/A"
            print(f"       MD={mds}s, MDP={mdps}s, P12={p12s}s")
            print(
                f"       Obj: MD={mdos}  MDP={mdpos}  P12={p12os}  |  {status}"
            )


# ---------------------------------------------------------------------------
# Unit tests: build_instance
# ---------------------------------------------------------------------------

def test_build_instance_shape():
    """build_instance returns W and D of size n×n."""
    W, D = build_instance(5, 2, seed=42)
    assert len(W) == 5 and len(W[0]) == 5
    assert len(D) == 5 and len(D[0]) == 5


def test_build_instance_diagonal_zero():
    """W and D have zero diagonal."""
    W, D = build_instance(4, 2, seed=43)
    for i in range(4):
        assert W[i][i] == 0
        assert D[i][i] == 0


def test_build_instance_reproducibility():
    """Same seed gives same W and D."""
    W1, D1 = build_instance(5, 2, seed=100)
    W2, D2 = build_instance(5, 2, seed=100)
    assert W1 == W2 and D1 == D2


def test_build_instance_different_seeds():
    """Different seeds give different data."""
    W1, D1 = build_instance(5, 2, seed=1)
    W2, D2 = build_instance(5, 2, seed=2)
    assert W1 != W2 or D1 != D2


def test_build_instance_no_negative():
    """W and D have non-negative entries (after zero diag)."""
    W, D = build_instance(6, 3, seed=44)
    for i in range(6):
        for j in range(6):
            assert W[i][j] >= 0 and D[i][j] >= 0


# ---------------------------------------------------------------------------
# Unit tests: preprocess
# ---------------------------------------------------------------------------

def test_preprocess_returns_all_keys():
    """preprocess returns H, C, L, K, cost_map, arcs_sorted, od_pairs."""
    W, D = build_instance(4, 2, seed=45)
    H, C, L, K, cost_map, arcs_sorted, od_pairs = preprocess(4, W, D)
    assert H is not None and C is not None and L is not None
    assert K is not None and cost_map is not None and arcs_sorted is not None
    assert od_pairs is not None


def test_preprocess_H_size():
    """H has n*(n-1) arcs."""
    W, D = build_instance(5, 2, seed=46)
    H, *_ = preprocess(5, W, D)
    assert len(H) == 5 * 4


def test_preprocess_K_ij_positive():
    """K[(i,j)] >= 1 for i != j."""
    W, D = build_instance(4, 2, seed=47)
    _, _, _, K, *_ = preprocess(4, W, D)
    for (i, j), k in K.items():
        assert i != j
        assert k >= 1


# ---------------------------------------------------------------------------
# Unit tests: F3 (hub_arc_models)
# ---------------------------------------------------------------------------

def test_f3_fixed_3_2_optimal():
    """F3 solves fixed n=3, p=2 instance to optimality."""
    W1 = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D1 = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    res = solve_hub_arc_F3(3, 2, W1, D1, gurobi_output=False)
    assert res["status"] == "OPTIMAL"
    assert res["objective"] is not None and res["objective"] >= 0


def test_f3_fixed_4_2_optimal():
    """F3 solves fixed n=4, p=2 instance."""
    W2 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D2 = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    res = solve_hub_arc_F3(4, 2, W2, D2, gurobi_output=False)
    assert res["status"] == "OPTIMAL"


def test_f3_returns_time():
    """F3 result includes time."""
    W, D = build_instance(4, 2, seed=48)
    res = solve_hub_arc_F3(4, 2, W, D, gurobi_output=False)
    assert "time" in res and res["time"] >= 0


# ---------------------------------------------------------------------------
# Unit tests: run_instance and aggregate
# ---------------------------------------------------------------------------

def test_run_instance_returns_all_keys():
    """run_instance returns n, p, f3_obj, norm_obj, new_obj, match, times, statuses."""
    W, D = build_instance(4, 2, seed=49)
    r = run_instance(4, 2, W, D, time_limit=30.0)
    for key in ("n", "p", "f3_obj", "f3_time", "norm_obj", "new_obj", "match"):
        assert key in r


def test_run_instance_fixed_3_2_match():
    """F3, Norm Benders, New Benders objectives match on fixed n=3, p=2."""
    W1 = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D1 = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    r = run_instance(3, 2, W1, D1, time_limit=60.0)
    assert r["match"], f"f3={r['f3_obj']} norm={r['norm_obj']} new={r['new_obj']}"


def test_run_instance_fixed_4_2_match():
    """Objectives match on fixed n=4, p=2."""
    W2 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D2 = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    r = run_instance(4, 2, W2, D2, time_limit=60.0)
    assert r["match"]


def test_aggregate_by_instance_structure():
    """aggregate_by_instance returns dict keyed by (n, p) with count, means, match_count."""
    W, D = build_instance(4, 2, seed=50)
    results = [run_instance(4, 2, W, D, time_limit=20.0)]
    agg = aggregate_by_instance(results)
    assert (4, 2) in agg
    a = agg[(4, 2)]
    assert "count" in a and "f3_mean" in a and "match_count" in a
    assert a["count"] == 1


def _fmt_t(x: Optional[float]) -> str:
    return f"{x:.4f}" if x is not None else "N/A"


def test_run_instance_n10_md_and_phase12_smoke():
    """
    Small instance (n=10): run full chain including MD, MD Pareto, and Pareto phase12.
    Checks the extra keys exist and all objectives agree (match) when solvers finish.
    Prints time (s) and objective per solver (use: pytest -s to see output).
    """
    n, p, seed = 10, 4, 123
    W, D = build_instance(n, p, seed)
    r = run_instance(
        n,
        p,
        W,
        D,
        time_limit=180.0,
        include_md_and_phase12=True,
    )
    for k in (
        "md_time",
        "md_obj",
        "md_status",
        "mdp_time",
        "mdp_obj",
        "mdp_status",
        "p12_time",
        "p12_obj",
        "p12_status",
    ):
        assert k in r, f"missing key {k}"
    assert r["f3_obj"] is not None
    assert r["norm_obj"] is not None
    assert r["new_obj"] is not None
    assert r["md_obj"] is not None
    assert r["mdp_obj"] is not None
    assert r["p12_obj"] is not None

    # Echo results (time in seconds, objective) — visible with: pytest -s ... 
    print(
        f"\n=== test_run_instance_n10_md_and_phase12_smoke n={n} p={p} seed={seed} ===\n"
        f"  {'Solver':<14}  {'time (s)':>12}  {'objective':>16}  status\n"
        f"  {'F3':<14}  {_fmt_t(r.get('f3_time')):>12}  "
        f"{r.get('f3_obj')!r:>16}  {r.get('f3_status')!r}\n"
        f"  {'Norm Benders':<14}  {_fmt_t(r.get('norm_time')):>12}  "
        f"{r.get('norm_obj')!r:>16}  {r.get('norm_status')!r}\n"
        f"  {'New Benders':<14}  {_fmt_t(r.get('new_time')):>12}  "
        f"{r.get('new_obj')!r:>16}  {r.get('new_status')!r}\n"
        f"  {'MD Benders':<14}  {_fmt_t(r.get('md_time')):>12}  "
        f"{r.get('md_obj')!r:>16}  {r.get('md_status')!r}\n"
        f"  {'MD Pareto':<14}  {_fmt_t(r.get('mdp_time')):>12}  "
        f"{r.get('mdp_obj')!r:>16}  {r.get('mdp_status')!r}\n"
        f"  {'Pareto P12':<14}  {_fmt_t(r.get('p12_time')):>12}  "
        f"{r.get('p12_obj')!r:>16}  {r.get('p12_status')!r}\n"
        f"  match (all objs within 1e-4): {r.get('match')!r}\n"
        f"==============================================\n",
        flush=True,
    )

    assert r["match"], (
        f"n={n} p={p} seed={seed} objectives should agree: "
        f"f3={r['f3_obj']} norm={r['norm_obj']} new={r['new_obj']} "
        f"md={r['md_obj']} mdp={r['mdp_obj']} p12={r['p12_obj']}"
    )


# ---------------------------------------------------------------------------
# Integration tests: F3 vs Benders objective match (many seeds/sizes)
# ---------------------------------------------------------------------------

def test_match_n3_p2_seed_42():
    W, D = build_instance(3, 2, seed=42)
    assert run_instance(3, 2, W, D, time_limit=60.0)["match"]


def test_match_n4_p2_seed_43():
    W, D = build_instance(4, 2, seed=43)
    assert run_instance(4, 2, W, D, time_limit=60.0)["match"]


def test_match_n4_p3_seed_44():
    W, D = build_instance(4, 3, seed=44)
    assert run_instance(4, 3, W, D, time_limit=60.0)["match"]


def test_match_n5_p2_seed_45():
    W, D = build_instance(5, 2, seed=45)
    assert run_instance(5, 2, W, D, time_limit=90.0)["match"]


def test_match_n5_p3_seed_46():
    W, D = build_instance(5, 3, seed=46)
    assert run_instance(5, 3, W, D, time_limit=90.0)["match"]


def test_match_n5_p2_seed_47():
    W, D = build_instance(5, 2, seed=47)
    assert run_instance(5, 2, W, D, time_limit=90.0)["match"]


def test_match_n6_p3_seed_48():
    W, D = build_instance(6, 3, seed=48)
    assert run_instance(6, 3, W, D, time_limit=120.0)["match"]


def test_match_n6_p4_seed_49():
    W, D = build_instance(6, 4, seed=49)
    assert run_instance(6, 4, W, D, time_limit=120.0)["match"]


def test_match_n5_p2_seed_50():
    W, D = build_instance(5, 2, seed=50)
    assert run_instance(5, 2, W, D, time_limit=60.0)["match"]


def test_match_n4_p2_seed_51():
    W, D = build_instance(4, 2, seed=51)
    assert run_instance(4, 2, W, D, time_limit=60.0)["match"]


def test_match_n3_p3_fixed():
    """Fixed instance n=3, p=3."""
    W1 = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D1 = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    assert run_instance(3, 3, W1, D1, time_limit=60.0)["match"]


def test_match_n4_p3_fixed():
    """Fixed instance n=4, p=3."""
    W2 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D2 = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    assert run_instance(4, 3, W2, D2, time_limit=60.0)["match"]


def test_new_benders_use_phase1_false():
    """New Benders runs with use_phase1=False and returns consistent result."""
    W, D = build_instance(4, 2, seed=52)
    r = run_instance(4, 2, W, D, use_phase1=False, time_limit=60.0)
    assert "new_obj" in r and "new_status" in r
    assert r["new_obj"] is not None or r["new_status"] != "OPTIMAL"


def test_run_instance_time_limit_short():
    """run_instance respects short time_limit (no crash)."""
    W, D = build_instance(6, 4, seed=53)
    r = run_instance(6, 4, W, D, time_limit=1.0)
    assert "f3_time" in r and "match" in r


def test_f3_objective_non_negative():
    """F3 objective is non-negative for random instance."""
    W, D = build_instance(4, 2, seed=54)
    res = solve_hub_arc_F3(4, 2, W, D, gurobi_output=False)
    assert res["objective"] is None or res["objective"] >= 0


def test_run_instance_n7_p4_seed_55():
    W, D = build_instance(7, 4, seed=55)
    assert run_instance(7, 4, W, D, time_limit=120.0)["match"]


def test_run_instance_n5_p2_seed_56():
    W, D = build_instance(5, 2, seed=56)
    assert run_instance(5, 2, W, D, time_limit=60.0)["match"]


def test_run_instance_n6_p3_seed_57():
    W, D = build_instance(6, 3, seed=57)
    assert run_instance(6, 3, W, D, time_limit=90.0)["match"]


def test_run_instance_n4_p2_seed_58():
    W, D = build_instance(4, 2, seed=58)
    assert run_instance(4, 2, W, D, time_limit=60.0)["match"]


def test_run_instance_n5_p3_seed_59():
    W, D = build_instance(5, 3, seed=59)
    assert run_instance(5, 3, W, D, time_limit=90.0)["match"]


def test_run_instance_n6_p4_seed_60():
    W, D = build_instance(6, 4, seed=60)
    assert run_instance(6, 4, W, D, time_limit=120.0)["match"]


# ---------------------------------------------------------------------------
# Large instance tests: n from 10 to 50 (with time_limit for scalability)
# ---------------------------------------------------------------------------

def test_match_n10_p4_seed_70():
    W, D = build_instance(10, 4, seed=70)
    assert run_instance(10, 4, W, D, time_limit=180.0)["match"]


def test_match_n10_p5_seed_71():
    W, D = build_instance(10, 5, seed=71)
    assert run_instance(10, 5, W, D, time_limit=180.0)["match"]


def test_match_n10_p6_seed_72():
    W, D = build_instance(10, 6, seed=72)
    assert run_instance(10, 6, W, D, time_limit=180.0)["match"]


def test_match_n12_p5_seed_73():
    W, D = build_instance(12, 5, seed=73)
    assert run_instance(12, 5, W, D, time_limit=200.0)["match"]


def test_match_n12_p6_seed_74():
    W, D = build_instance(12, 6, seed=74)
    assert run_instance(12, 6, W, D, time_limit=200.0)["match"]


def test_match_n15_p6_seed_75():
    W, D = build_instance(15, 6, seed=75)
    assert run_instance(15, 6, W, D, time_limit=240.0)["match"]


def test_match_n15_p8_seed_76():
    W, D = build_instance(15, 8, seed=76)
    assert run_instance(15, 8, W, D, time_limit=240.0)["match"]


def test_match_n20_p8_seed_77():
    W, D = build_instance(20, 8, seed=77)
    assert run_instance(20, 8, W, D, time_limit=300.0)["match"]


def test_match_n20_p10_seed_78():
    W, D = build_instance(20, 10, seed=78)
    assert run_instance(20, 10, W, D, time_limit=300.0)["match"]


def test_match_n25_p10_seed_79():
    W, D = build_instance(25, 10, seed=79)
    assert run_instance(25, 10, W, D, time_limit=360.0)["match"]


def test_match_n25_p12_seed_80():
    W, D = build_instance(25, 12, seed=80)
    assert run_instance(25, 12, W, D, time_limit=360.0)["match"]


def test_match_n30_p12_seed_81():
    W, D = build_instance(30, 12, seed=81)
    assert run_instance(30, 12, W, D, time_limit=400.0)["match"]


def test_match_n30_p15_seed_82():
    W, D = build_instance(30, 15, seed=82)
    assert run_instance(30, 15, W, D, time_limit=400.0)["match"]


def test_match_n35_p14_seed_83():
    W, D = build_instance(35, 14, seed=83)
    assert run_instance(35, 14, W, D, time_limit=480.0)["match"]


def test_match_n35_p18_seed_84():
    W, D = build_instance(35, 18, seed=84)
    assert run_instance(35, 18, W, D, time_limit=480.0)["match"]


def test_match_n40_p16_seed_85():
    W, D = build_instance(40, 16, seed=85)
    assert run_instance(40, 16, W, D, time_limit=600.0)["match"]


def test_match_n40_p20_seed_86():
    W, D = build_instance(40, 20, seed=86)
    assert run_instance(40, 20, W, D, time_limit=600.0)["match"]


def test_match_n45_p18_seed_87():
    W, D = build_instance(45, 18, seed=87)
    assert run_instance(45, 18, W, D, time_limit=600.0)["match"]


def test_match_n45_p22_seed_88():
    W, D = build_instance(45, 22, seed=88)
    assert run_instance(45, 22, W, D, time_limit=600.0)["match"]


def test_match_n50_p20_seed_89():
    W, D = build_instance(50, 20, seed=89)
    assert run_instance(50, 20, W, D, time_limit=600.0)["match"]


def test_match_n50_p25_seed_90():
    W, D = build_instance(50, 25, seed=90)
    assert run_instance(50, 25, W, D, time_limit=600.0)["match"]


def test_match_n10_p4_seed_91():
    W, D = build_instance(10, 4, seed=91)
    assert run_instance(10, 4, W, D, time_limit=120.0)["match"]


def test_match_n18_p8_seed_92():
    W, D = build_instance(18, 8, seed=92)
    assert run_instance(18, 8, W, D, time_limit=280.0)["match"]


def test_match_n22_p10_seed_93():
    W, D = build_instance(22, 10, seed=93)
    assert run_instance(22, 10, W, D, time_limit=320.0)["match"]


# Instance specs for script mode: n from 3 to 50 (fixed + random with seeds)
def get_script_instance_specs():
    """Return instance specs for main() script: small fixed + n=5..50 random."""
    W1 = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D1 = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    W2 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D2 = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    return [
        # Fixed (small, deterministic)
        {"n": 3, "p": 2, "fixed": True, "W": W1, "D": D1},
        {"n": 3, "p": 3, "fixed": True, "W": W1, "D": D1},
        {"n": 4, "p": 2, "fixed": True, "W": W2, "D": D2},
        {"n": 4, "p": 3, "fixed": True, "W": W2, "D": D2},
        # Random n=5..8
        {"n": 5, "p": 2, "fixed": False},
        {"n": 5, "p": 3, "fixed": False},
        {"n": 6, "p": 3, "fixed": False},
        {"n": 6, "p": 4, "fixed": False},
        {"n": 7, "p": 4, "fixed": False},
        {"n": 8, "p": 4, "fixed": False},
        {"n": 8, "p": 5, "fixed": False},
        # n=10 to n=50 (script mode: 1 seed each for speed; pytest runs full seeds)
        {"n": 10, "p": 4, "fixed": False},
        {"n": 10, "p": 5, "fixed": False},
        {"n": 10, "p": 6, "fixed": False},
        {"n": 12, "p": 5, "fixed": False},
        {"n": 12, "p": 6, "fixed": False},
        {"n": 15, "p": 6, "fixed": False},
        {"n": 15, "p": 8, "fixed": False},
        {"n": 18, "p": 8, "fixed": False},
        {"n": 20, "p": 8, "fixed": False},
        {"n": 20, "p": 10, "fixed": False},
        {"n": 22, "p": 10, "fixed": False},
        {"n": 25, "p": 10, "fixed": False},
        {"n": 25, "p": 12, "fixed": False},
        {"n": 30, "p": 12, "fixed": False},
        {"n": 30, "p": 15, "fixed": False},
        {"n": 35, "p": 14, "fixed": False},
        {"n": 35, "p": 18, "fixed": False},
        {"n": 40, "p": 16, "fixed": False},
        {"n": 40, "p": 20, "fixed": False},
        {"n": 45, "p": 18, "fixed": False},
        {"n": 45, "p": 22, "fixed": False},
        {"n": 50, "p": 20, "fixed": False},
        {"n": 50, "p": 25, "fixed": False},
    ]


def main():
    print("\n" + "=" * 95)
    print("COMPREHENSIVE TIMING TEST: p-Hub-Arc (F3 vs Normal Benders vs New Benders)")
    print("=" * 95)

    instance_specs = get_script_instance_specs()
    # Script mode: 1 seed per random config for faster run (n=3..50). For 5 seeds, set seeds = [42,43,44,45,46].
    seeds = [42]
    time_limit = 300.0  # allow larger n (up to 50) to complete

    n_fixed = sum(1 for s in instance_specs if s.get("fixed"))
    n_random = len(instance_specs) - n_fixed
    total_runs = n_fixed * 1 + n_random * len(seeds)

    print(f"\nInstance specs: {len(instance_specs)} configurations (n=3..50)")
    print(f"Seeds per random config: {len(seeds)} (or 1 for fixed) -> {total_runs} total runs")
    print(f"Time limit per solve: {time_limit}s")
    print("Full pytest suite (all tests): python -m pytest p-hub-arc/test_timing_comprehensive.py -v")
    print("\nRunning suite...")
    print("-" * 60)

    results = run_timing_suite(
        instance_specs,
        seeds=seeds,
        time_limit=time_limit,
        use_phase1=True,
        verbose=True,
    )

    agg = aggregate_by_instance(results)
    print_timing_table(results, agg)
    print_detailed_results(results)

    # Summary
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    total = len([r for r in results if "error" not in r])
    matches = sum(1 for r in results if r.get("match"))
    f3_wins = norm_wins = new_wins = 0
    for r in results:
        if "error" in r:
            continue
        times = [("F3", r.get("f3_time")), ("NormB", r.get("norm_time")), ("NewB", r.get("new_time"))]
        valid = [(n, t) for n, t in times if t is not None]
        if valid:
            fastest = min(valid, key=lambda x: x[1])[0]
            if fastest == "F3":
                f3_wins += 1
            elif fastest == "NormB":
                norm_wins += 1
            else:
                new_wins += 1

    print(f"Total runs:       {total}")
    print(f"Objectives match: {matches}/{total}")
    print(f"F3 fastest:       {f3_wins} runs")
    print(f"Normal Benders fastest: {norm_wins} runs")
    print(f"New Benders fastest:    {new_wins} runs")

    ok = matches == total
    print("\n" + ("All objectives matched." if ok else "WARNING: Some objectives did not match."))
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
