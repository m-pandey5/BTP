"""
test_timing_n60_plus.py

Same as test_timing_large.py but:
  1. Starts from n=60 (no n=30 sanity smoke test).
  2. Each solver (F3, NormB, NewB, MD, MDP, P12) is wrapped in its own
     try/except, so an F3 OOM crash does NOT abort the remaining five
     decomposition solvers for that run.

Usage:
  python3 test_timing_n60_plus.py

Skip F3 entirely (saves ~45 min of futile presolve per run):
  LARGE_BENCH_SKIP_F3=1 python3 test_timing_n60_plus.py

Quick mode (NormB + NewB only, no MD/MDP/P12):
  LARGE_BENCH_NO_EXTRA=1 python3 test_timing_n60_plus.py

Environment variables (same as test_timing_large.py):
  LARGE_BENCH_SKIP_F3=1          skip F3 direct MIP
  LARGE_BENCH_NO_EXTRA=1         skip MD, MD Pareto, Pareto P12
  LARGE_BENCH_NO_GUROBI_LOGS=1   suppress Gurobi solver output
  LARGE_BENCH_FORCE_TIME_LIMIT=N override per-solver time limit (seconds)
  LARGE_BENCH_LARGE_SEEDS=42,43  seeds for n>=100 (default: 42)
  LARGE_BENCH_ALL_SEEDS=1        use all seeds for every n
  LARGE_BENCH_MAX_RUNS=N         stop after N runs (debug)
"""

import csv
import os
import sys
import time
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    print(
        "Error: numpy is required. Activate your venv and run:\n"
        "  python3 -m pip install numpy",
        file=sys.stderr,
    )
    sys.exit(1)

from hub_arc_models import solve_hub_arc_F3, HubArcBenders
from new_model_hub_arc import solve_benders_hub_arc, preprocess
from md_benders_hub_arc import solve_md_benders_hub_arc
from md_benders_hub_arc_pareto import solve_md_benders_hub_arc_pareto
from new_model_hub_arc_pareto_phase12 import solve_benders_hub_arc_pareto_phase12
from test_timing_comprehensive import build_instance

GRB_OPTIMAL_STATUS = {2, "OPTIMAL"}


# ---------------------------------------------------------------------------
# Instance configurations — n=60 and above only
# ---------------------------------------------------------------------------

def get_n60_plus_specs() -> List[Dict]:
    """Instance configs starting from n=60 (two p values per n)."""
    configs = []
    for n, p1, p2 in [
        (60,  12, 15),
        (75,  15, 19),
        (100, 20, 25),
        (125, 25, 31),
        (150, 30, 38),
        (200, 40, 50),
    ]:
        configs.append({"n": n, "p": p1})
        configs.append({"n": n, "p": p2})
    return configs


# ---------------------------------------------------------------------------
# Per-solver independent runner — each solver catches its own exceptions
# ---------------------------------------------------------------------------

@contextmanager
def _heartbeat(label: str, interval_sec: float):
    if interval_sec <= 0:
        yield
        return
    stop = threading.Event()
    t0 = time.time()

    def _loop():
        n_hb = 0
        while not stop.wait(interval_sec):
            n_hb += 1
            print(
                f"\n      [running] {label}  (heartbeat #{n_hb}, ~{int(time.time()-t0)}s elapsed)",
                flush=True,
            )

    th = threading.Thread(target=_loop, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=1.0)


def _error_result(err: Exception) -> Dict[str, Any]:
    return {
        "objective": None,
        "time": None,
        "status": f"ERROR: {err}",
        "mip_gap": None,
        "error": str(err),
    }


def run_instance_independent(
    n: int,
    p: int,
    W: List,
    D: List,
    time_limit: float,
    skip_f3: bool,
    include_extras: bool,
    gurobi_logs: bool,
    heartbeat_sec: float,
) -> Dict[str, Any]:
    """
    Run all requested solvers independently.
    Each solver is wrapped in its own try/except — a crash in F3 (e.g. OOM)
    does NOT prevent NormB, NewB, MD, MDP, P12 from running.
    """
    hb = heartbeat_sec > 0

    # ---- F3 ----
    if skip_f3:
        f3_res: Dict[str, Any] = {"objective": None, "time": None, "status": "SKIPPED", "mip_gap": None}
        print("(F3 skipped) | ", end="", flush=True)
    else:
        print("(F3 …)", end=" ", flush=True)
        try:
            with _heartbeat("F3 direct MIP", heartbeat_sec) if hb else _heartbeat("", 0):
                f3_res = solve_hub_arc_F3(n, p, W, D, gurobi_output=gurobi_logs, time_limit=time_limit)
            print(f"done {f3_res.get('time') or 0.0:.2f}s | ", end="", flush=True)
        except Exception as e:
            print(f"ERROR({e}) | ", end="", flush=True)
            f3_res = _error_result(e)

    # ---- Normal Benders ----
    print("(Norm Benders …)", end=" ", flush=True)
    try:
        b_norm = HubArcBenders(n=n, p=p, W=W, D=D, verbose=gurobi_logs)
        with _heartbeat("Normal Benders", heartbeat_sec) if hb else _heartbeat("", 0):
            norm_res = b_norm.solve(time_limit=time_limit)
        # apply constant correction (same logic as test_timing_comprehensive.py)
        _, C, _, K, _, _, _ = preprocess(n, W, D)
        if norm_res["objective"] is not None:
            add_const = sum(C[(i, j)][0] for (i, j) in K if i != j and K[(i, j)] == 1)
            norm_res = dict(norm_res, objective=norm_res["objective"] + add_const)
        print(f"done {norm_res.get('time') or 0.0:.2f}s | ", end="", flush=True)
    except Exception as e:
        print(f"ERROR({e}) | ", end="", flush=True)
        norm_res = _error_result(e)

    # ---- New Benders ----
    print("(New Benders …)", end=" ", flush=True)
    try:
        with _heartbeat("New Benders (Phase1+Phase2)", heartbeat_sec) if hb else _heartbeat("", 0):
            new_res = solve_benders_hub_arc(
                n, p, W, D, verbose=gurobi_logs, use_phase1=True, time_limit=time_limit
            )
        suffix = " | " if include_extras else ""
        print(f"done {new_res.get('time') or 0.0:.2f}s{suffix}", end="", flush=True)
    except Exception as e:
        suffix = " | " if include_extras else ""
        print(f"ERROR({e}){suffix}", end="", flush=True)
        new_res = _error_result(e)

    md_res: Dict[str, Any] = {"objective": None, "time": None, "status": "SKIPPED", "mip_gap": None}
    mdp_res: Dict[str, Any] = {"objective": None, "time": None, "status": "SKIPPED", "mip_gap": None}
    p12_res: Dict[str, Any] = {"objective": None, "time": None, "status": "SKIPPED", "mip_gap": None}

    if include_extras:
        # ---- MD Benders ----
        print("(MD …)", end=" ", flush=True)
        try:
            with _heartbeat("MD Benders (McDaniel-Devine)", heartbeat_sec) if hb else _heartbeat("", 0):
                md_res = solve_md_benders_hub_arc(
                    n, p, W, D, time_limit=time_limit, verbose=gurobi_logs, use_phase1=True
                )
            print(f"done {md_res.get('time') or 0.0:.2f}s | ", end="", flush=True)
        except Exception as e:
            print(f"ERROR({e}) | ", end="", flush=True)
            md_res = _error_result(e)

        # ---- MD Pareto ----
        print("(MD Pareto …)", end=" ", flush=True)
        try:
            with _heartbeat("MD Pareto Benders (two_step)", heartbeat_sec) if hb else _heartbeat("", 0):
                mdp_res = solve_md_benders_hub_arc_pareto(
                    n, p, W, D, time_limit=time_limit, verbose=gurobi_logs, use_phase1=True
                )
            print(f"done {mdp_res.get('time') or 0.0:.2f}s | ", end="", flush=True)
        except Exception as e:
            print(f"ERROR({e}) | ", end="", flush=True)
            mdp_res = _error_result(e)

        # ---- Pareto Phase12 ----
        print("(P12 …)", end=" ", flush=True)
        try:
            with _heartbeat("Pareto Phase12 (two_step)", heartbeat_sec) if hb else _heartbeat("", 0):
                p12_res = solve_benders_hub_arc_pareto_phase12(
                    n, p, W, D, time_limit=time_limit, verbose=gurobi_logs,
                    use_phase1=True, pareto_method="two_step",
                )
            print(f"done {p12_res.get('time') or 0.0:.2f}s", flush=True)
        except Exception as e:
            print(f"ERROR({e})", flush=True)
            p12_res = _error_result(e)
    else:
        print(flush=True)

    # ---- match check (only non-None, non-error objectives) ----
    objs = [
        r.get("objective")
        for r in [f3_res, norm_res, new_res, md_res, mdp_res, p12_res]
        if r.get("objective") is not None and "error" not in r
    ]
    match = len(objs) > 1 and (max(objs) - min(objs) < 1e-4)

    return {
        "n": n,
        "p": p,
        "f3_time":    f3_res.get("time"),
        "f3_obj":     f3_res.get("objective"),
        "f3_status":  str(f3_res.get("status", "")),
        "f3_gap":     f3_res.get("mip_gap"),
        "norm_time":  norm_res.get("time"),
        "norm_obj":   norm_res.get("objective"),
        "norm_status": str(norm_res.get("status", "")),
        "norm_gap":   norm_res.get("mip_gap"),
        "new_time":   new_res.get("time"),
        "new_obj":    new_res.get("objective"),
        "new_status": str(new_res.get("status", "")),
        "new_gap":    new_res.get("mip_gap"),
        "md_time":    md_res.get("time"),
        "md_obj":     md_res.get("objective"),
        "md_status":  str(md_res.get("status", "")),
        "md_gap":     md_res.get("mip_gap"),
        "mdp_time":   mdp_res.get("time"),
        "mdp_obj":    mdp_res.get("objective"),
        "mdp_status": str(mdp_res.get("status", "")),
        "mdp_gap":    mdp_res.get("mip_gap"),
        "p12_time":   p12_res.get("time"),
        "p12_obj":    p12_res.get("objective"),
        "p12_status": str(p12_res.get("status", "")),
        "p12_gap":    p12_res.get("mip_gap"),
        "match":      match,
    }


# ---------------------------------------------------------------------------
# Helpers (same as test_timing_large.py)
# ---------------------------------------------------------------------------

def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "y")


def _parse_seed_list(value: str) -> List[int]:
    return [int(t.strip()) for t in value.split(",") if t.strip()]


def _seeds_for_n(n: int, default_seeds: List[int], large_seeds: List[int], all_seeds: bool) -> List[int]:
    if all_seeds or n <= 75:
        return list(default_seeds)
    return list(large_seeds)


def _effective_time_limit(n: int, forced: Optional[float]) -> float:
    if forced is not None:
        return forced
    if n in (60, 75):
        return 7200.0
    return 10800.0


def _fmt_time(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def _fmt_obj(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "N/A"


def _fastest_winner(r: Dict, skip_f3: bool, include_extras: bool) -> Optional[str]:
    candidates: List[Tuple[str, float]] = []
    if not skip_f3 and r.get("f3_time") is not None and r["f3_time"] >= 0:
        candidates.append(("F3", r["f3_time"]))
    for name, key in [("NormB", "norm_time"), ("NewB", "new_time")]:
        t = r.get(key)
        if t is not None and t >= 0:
            candidates.append((name, t))
    if include_extras:
        for name, key in [("MD", "md_time"), ("MDP", "mdp_time"), ("P12", "p12_time")]:
            t = r.get(key)
            if t is not None and t >= 0:
                candidates.append((name, t))
    return min(candidates, key=lambda x: x[1])[0] if candidates else None


def save_results_csv(results: List[Dict], path: str) -> None:
    fieldnames = [
        "n", "p", "seed", "effective_time_limit",
        "f3_time",   "f3_obj",   "f3_status",   "f3_gap",
        "norm_time", "norm_obj", "norm_status", "norm_gap",
        "new_time",  "new_obj",  "new_status",  "new_gap",
        "md_time",   "md_obj",   "md_status",   "md_gap",
        "mdp_time",  "mdp_obj",  "mdp_status",  "mdp_gap",
        "p12_time",  "p12_obj",  "p12_status",  "p12_gap",
        "match",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"\nResults saved to: {path}")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


@contextmanager
def _tee_output(log_path: str):
    old_out, old_err = sys.stdout, sys.stderr
    with open(log_path, "a", buffering=1) as log_f:
        sys.stdout = _Tee(old_out, log_f)
        sys.stderr = _Tee(old_err, log_f)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = old_out, old_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    skip_f3      = _env_truthy("LARGE_BENCH_SKIP_F3")
    include_extras = not _env_truthy("LARGE_BENCH_NO_EXTRA")
    gurobi_logs  = not _env_truthy("LARGE_BENCH_NO_GUROBI_LOGS")

    default_seeds = [42, 43, 44]
    large_seeds   = _parse_seed_list(os.environ.get("LARGE_BENCH_LARGE_SEEDS", "42"))
    all_seeds     = _env_truthy("LARGE_BENCH_ALL_SEEDS")
    force_limit   = float(os.environ["LARGE_BENCH_FORCE_TIME_LIMIT"]) \
        if os.environ.get("LARGE_BENCH_FORCE_TIME_LIMIT", "").strip() else None
    max_runs_env  = os.environ.get("LARGE_BENCH_MAX_RUNS", "").strip()
    max_runs      = int(max_runs_env) if max_runs_env else None

    instance_specs = get_n60_plus_specs()
    total_runs = sum(
        len(_seeds_for_n(s["n"], default_seeds, large_seeds, all_seeds))
        for s in instance_specs
    )

    run_ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir   = os.path.join(_this_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    log_path   = os.path.join(runs_dir, f"test_timing_n60plus_{run_ts}.log")
    csv_live   = os.path.join(runs_dir, f"n60plus_results_live_{run_ts}.csv")

    with _tee_output(log_path):
        print("\n" + "=" * 95)
        print("N60+ BATCH: Solvers run INDEPENDENTLY per instance (F3 crash does not abort others)")
        print("Configs: n=60,75,100,125,150,200  |  Two p values per n  |  No n=30 smoke test")
        print("=" * 95)
        print(f"\nTotal planned runs : {total_runs}")
        print(f"Default seeds      : {default_seeds}  |  Large-n seeds: {large_seeds}")
        print(f"Time limit policy  : n in {{60,75}}: 7200s  |  n>=100: 10800s")
        if force_limit is not None:
            print(f"Forced time limit  : {force_limit}s  (LARGE_BENCH_FORCE_TIME_LIMIT)")
        if skip_f3:
            print("F3: SKIPPED  (LARGE_BENCH_SKIP_F3=1) — saves ~45 min of futile presolve per run")
        else:
            print("F3: enabled  (will attempt; OOM caught independently, won't abort other solvers)")
        if not include_extras:
            print("MD / MD Pareto / P12: SKIPPED  (LARGE_BENCH_NO_EXTRA=1)")
        print(f"Gurobi logs: {'enabled' if gurobi_logs else 'disabled'}")
        print(f"Run log: {log_path}")
        print(f"Live CSV: {csv_live}")
        print("\nRunning suite...")
        print("-" * 60)

        results: List[Dict] = []
        completed = 0
        run_idx   = 0
        stop_early = False

        for spec in instance_specs:
            n, p = spec["n"], spec["p"]
            for seed in _seeds_for_n(n, default_seeds, large_seeds, all_seeds):
                run_idx += 1
                tl = _effective_time_limit(n, force_limit)
                label = f"[{run_idx}] n={n}, p={p}, seed={seed}"
                print(f"  {label} ...", end=" ", flush=True)

                W, D = build_instance(n, p, seed)

                r = run_instance_independent(
                    n=n, p=p, W=W, D=D,
                    time_limit=tl,
                    skip_f3=skip_f3,
                    include_extras=include_extras,
                    gurobi_logs=gurobi_logs,
                    heartbeat_sec=(0.0 if gurobi_logs else 20.0),
                )
                r["seed"]                 = seed
                r["effective_time_limit"] = tl

                fastest = _fastest_winner(r, skip_f3, include_extras)
                if include_extras:
                    print(
                        f"      F3={_fmt_time(r.get('f3_time'))}s  "
                        f"NormB={_fmt_time(r.get('norm_time'))}s  "
                        f"NewB={_fmt_time(r.get('new_time'))}s  "
                        f"MD={_fmt_time(r.get('md_time'))}s  "
                        f"MDP={_fmt_time(r.get('mdp_time'))}s  "
                        f"P12={_fmt_time(r.get('p12_time'))}s  "
                        f"(fastest: {fastest})"
                    )
                    print(
                        f"      Obj: "
                        f"F3={_fmt_obj(r.get('f3_obj'))}  "
                        f"NormB={_fmt_obj(r.get('norm_obj'))}  "
                        f"NewB={_fmt_obj(r.get('new_obj'))}  "
                        f"MD={_fmt_obj(r.get('md_obj'))}  "
                        f"MDP={_fmt_obj(r.get('mdp_obj'))}  "
                        f"P12={_fmt_obj(r.get('p12_obj'))}  |  "
                        f"{'Match' if r.get('match') else 'MISMATCH/INCOMPLETE'}"
                    )
                else:
                    print(
                        f"      F3={_fmt_time(r.get('f3_time'))}s  "
                        f"NormB={_fmt_time(r.get('norm_time'))}s  "
                        f"NewB={_fmt_time(r.get('new_time'))}s  "
                        f"(fastest: {fastest})"
                    )
                    print(
                        f"      Obj: "
                        f"F3={_fmt_obj(r.get('f3_obj'))}  "
                        f"NormB={_fmt_obj(r.get('norm_obj'))}  "
                        f"NewB={_fmt_obj(r.get('new_obj'))}  |  "
                        f"{'Match' if r.get('match') else 'MISMATCH/INCOMPLETE'}"
                    )

                # per-solver status warnings
                all_solvers = [
                    ("F3",    "f3_status"),
                    ("NormB", "norm_status"),
                    ("NewB",  "new_status"),
                ]
                if include_extras:
                    all_solvers += [
                        ("MD",  "md_status"),
                        ("MDP", "mdp_status"),
                        ("P12", "p12_status"),
                    ]
                for sname, skey in all_solvers:
                    st = str(r.get(skey, ""))
                    if st not in ("OPTIMAL", "SKIPPED", "2"):
                        print(f"      WARN {sname}: status={st}")

                results.append(r)
                completed += 1
                save_results_csv(results, csv_live)

                if max_runs is not None and completed >= max_runs:
                    stop_early = True
                    print(f"\nStopping early (LARGE_BENCH_MAX_RUNS={max_runs}).")
                    break
            if stop_early:
                break

        # ---- Final summary ----
        print("\n" + "=" * 95)
        print("SUMMARY")
        print("=" * 95)
        total   = len(results)
        matches = sum(1 for r in results if r.get("match"))
        wins    = {k: 0 for k in ["F3", "NormB", "NewB", "MD", "MDP", "P12"]}
        for r in results:
            w = _fastest_winner(r, skip_f3, include_extras)
            if w in wins:
                wins[w] += 1

        print(f"Total runs           : {total}")
        print(f"Objectives matched   : {matches}/{total}")
        for name, count in wins.items():
            if name == "F3" and skip_f3:
                continue
            if name in ("MD", "MDP", "P12") and not include_extras:
                continue
            print(f"{name:<6} fastest      : {count} runs")

        final_csv = os.path.join(_this_dir, "n60plus_results.csv")
        save_results_csv(results, final_csv)


if __name__ == "__main__":
    main()
