"""
Large-Instance Batch Timing: F3, Norm, New, plus optional MD and Pareto variants.

**Solvers (same random W, D per row; sequential, not parallel; each has its own
Gurobi TimeLimit, e.g. 600s per call — total wall time is the sum of all calls):**

  1) F3 (direct MIP) — skippable with LARGE_BENCH_SKIP_F3=1
  2) Normal Benders (HubArcBenders)
  3) New Benders (solve_benders_hub_arc, Phase1+2)
  4) McDaniel-Devine (md_benders_hub_arc) — unless LARGE_BENCH_NO_EXTRA=1
  5) MD + Pareto (md_benders_hub_arc_pareto, two_step)
  6) New-model Pareto phase12 (new_model_hub_arc_pareto_phase12, two_step)

**Quick mode (F3 + Norm + New only, no MD / MD Pareto / P12):**

  LARGE_BENCH_NO_EXTRA=1 python3 test_timing_large.py

**Skip F3 only** (e.g. large n, slow Python build):

  LARGE_BENCH_SKIP_F3=1 python3 test_timing_large.py

`match` in CSV: when extras run, all reported objectives (F3 if present, else
the five Benders) must agree within 1e-4.

Output: [large_instance_results.csv](large_instance_results.csv) in this folder
(one row per n,p,seed; all solver columns; failed cells empty/None).

Prerequisites: numpy, gurobipy. Run from p-hub-arc:  python3 test_timing_large.py
"""

import csv
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Tuple

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

# Fail fast with a clear message (import chain needs numpy)
try:
    import numpy  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    print(
        "Error: numpy is required. Activate your venv and run:\n"
        "  python3 -m pip install numpy",
        file=sys.stderr,
    )
    sys.exit(1)

from test_timing_comprehensive import (
    aggregate_by_instance,
    build_instance,
    print_detailed_results,
    print_timing_table,
    run_instance,
)


def get_large_instance_specs() -> List[Dict]:
    """Instance configs: one small n=30 smoke case first, then n=60..200 (two p each)."""
    configs = []
    # Smoke: runs first so you can confirm the script solves before the heavy n=60+ block
    configs.append({"n": 30, "p": 6, "fixed": False})
    for n, p1, p2 in [
        (60,  12, 15),
        (75,  15, 19),
        (100, 20, 25),
        (125, 25, 31),
        (150, 30, 38),
        (200, 40, 50),
    ]:
        configs.append({"n": n, "p": p1, "fixed": False})
        configs.append({"n": n, "p": p2, "fixed": False})
    return configs


def save_results_csv(results: List[Dict], path: str) -> None:
    """Save per-run results to a CSV (all solver time/obj/status columns + match)."""
    fieldnames = [
        "n", "p", "seed",
        "f3_time", "f3_obj", "f3_status",
        "f3_gap",
        "norm_time", "norm_obj", "norm_status",
        "norm_gap",
        "new_time", "new_obj", "new_status",
        "new_gap",
        "md_time", "md_obj", "md_status",
        "md_gap",
        "mdp_time", "mdp_obj", "mdp_status",
        "mdp_gap",
        "p12_time", "p12_obj", "p12_status",
        "p12_gap",
        "match",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"\nResults saved to: {path}")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "y")


def _fastest_winner(
    r: Dict, include_extras: bool, skip_f3: bool
) -> Optional[str]:
    if "error" in r:
        return None
    times: List[Tuple[str, float]] = []
    if not skip_f3 and r.get("f3_time") is not None and r["f3_time"] >= 0:
        times.append(("F3", r["f3_time"]))
    for name, key in [
        ("NormB", "norm_time"),
        ("NewB", "new_time"),
    ]:
        t = r.get(key)
        if t is not None and t >= 0:
            times.append((name, t))
    if include_extras:
        for name, key in [
            ("MD", "md_time"),
            ("MDP", "mdp_time"),
            ("P12", "p12_time"),
        ]:
            t = r.get(key)
            if t is not None and t >= 0:
                times.append((name, t))
    if not times:
        return None
    return min(times, key=lambda x: x[1])[0]


class _Tee:
    """Mirror writes to multiple file-like streams."""

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
    """Duplicate stdout/stderr to a run log file."""
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


def _write_live_progress(
    progress_path: str,
    completed_runs: int,
    total_runs: int,
    current_case: str,
    last_completed_case: Optional[str],
    live_csv_path: str,
    run_log_path: str,
) -> None:
    with open(progress_path, "w") as f:
        f.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"completed_runs: {completed_runs}/{total_runs}\n")
        f.write(f"current_case: {current_case}\n")
        f.write(f"last_completed_case: {last_completed_case or 'N/A'}\n")
        f.write(f"live_csv: {live_csv_path}\n")
        f.write(f"run_log: {run_log_path}\n")


def _fmt_time(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def _fmt_obj(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "N/A"


def _fmt_gap(v: Optional[float]) -> str:
    return f"{v:.6f}" if v is not None else "N/A"


def _print_solver_warnings(r: Dict, include_extras: bool, skip_f3: bool) -> None:
    solvers: List[Tuple[str, str, str, str]] = []
    if not skip_f3:
        solvers.append(("F3", "f3_status", "f3_obj", "f3_gap"))
    solvers.extend(
        [
            ("NormB", "norm_status", "norm_obj", "norm_gap"),
            ("NewB", "new_status", "new_obj", "new_gap"),
        ]
    )
    if include_extras:
        solvers.extend(
            [
                ("MD", "md_status", "md_obj", "md_gap"),
                ("MDP", "mdp_status", "mdp_obj", "mdp_gap"),
                ("P12", "p12_status", "p12_obj", "p12_gap"),
            ]
        )
    for name, status_key, obj_key, gap_key in solvers:
        status = str(r.get(status_key, "UNKNOWN"))
        if status in ("OPTIMAL", "SKIPPED"):
            continue
        print(
            f"      WARN {name}: status={status}, "
            f"incumbent={_fmt_obj(r.get(obj_key))}, gap={_fmt_gap(r.get(gap_key))}"
        )


def _print_gap_summary(r: Dict, include_extras: bool, skip_f3: bool) -> None:
    parts: List[str] = []
    if not skip_f3:
        parts.append(f"F3={_fmt_gap(r.get('f3_gap'))}")
    parts.append(f"NormB={_fmt_gap(r.get('norm_gap'))}")
    parts.append(f"NewB={_fmt_gap(r.get('new_gap'))}")
    if include_extras:
        parts.append(f"MD={_fmt_gap(r.get('md_gap'))}")
        parts.append(f"MDP={_fmt_gap(r.get('mdp_gap'))}")
        parts.append(f"P12={_fmt_gap(r.get('p12_gap'))}")
    print("      GAP SUMMARY: " + "  ".join(parts))


def main():
    skip_f3 = _env_truthy("LARGE_BENCH_SKIP_F3")
    # Default: run MD + MD Pareto + P12 after New. Set LARGE_BENCH_NO_EXTRA=1 to skip.
    include_extras = not _env_truthy("LARGE_BENCH_NO_EXTRA")
    # Default ON: show real Gurobi logs unless explicitly disabled.
    gurobi_logs = not _env_truthy("LARGE_BENCH_NO_GUROBI_LOGS")

    instance_specs = get_large_instance_specs()
    seeds = [42, 43, 44]
    time_limit = 600.0

    total_runs = len(instance_specs) * len(seeds)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = os.path.join(_this_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    run_log_path = os.path.join(runs_dir, f"test_timing_large_{run_ts}.log")
    live_csv_path = os.path.join(runs_dir, f"large_instance_results_live_{run_ts}.csv")
    progress_path = os.path.join(runs_dir, f"large_instance_progress_{run_ts}.txt")

    max_runs_env = os.environ.get("LARGE_BENCH_MAX_RUNS", "").strip()
    max_runs = int(max_runs_env) if max_runs_env else None

    with _tee_output(run_log_path):
        print("\n" + "=" * 95)
        if not skip_f3 and include_extras:
            print(
                "LARGE-INSTANCE BATCH: F3, Norm, New, then MD, MD Pareto, P12 (same W,D; "
                "per-solve time limit; wall time is sum of solvers)"
            )
        elif not skip_f3:
            print("LARGE-INSTANCE BATCH: F3, then Norm, then New (LARGE_BENCH_NO_EXTRA=1 — no MD/P12)")
        elif include_extras:
            print(
                "LARGE-INSTANCE BATCH: Norm, New, MD, MD Pareto, P12 (F3 SKIPPED; "
                "LARGE_BENCH_NO_EXTRA unset)"
            )
        else:
            print("LARGE-INSTANCE BATCH: Norm + New only (F3 and extras skipped)")
        print("=" * 95)

        print(
            f"\nInstance specs: {len(instance_specs)} configurations "
            f"(first: n=30 sanity, then n=60..200)"
        )
        print(f"Seeds per config: {seeds} -> {total_runs} total runs")
        print(f"Time limit per solve: {time_limit}s  (each solver call; not a shared cap)")
        print(f"Run log: {run_log_path}")
        print(f"Live CSV: {live_csv_path}")
        print(f"Progress: {progress_path}")
        if skip_f3:
            print("F3: skipped (unset LARGE_BENCH_SKIP_F3 to run F3)")
        if not include_extras:
            print("MD / MD Pareto / P12: skipped (unset LARGE_BENCH_NO_EXTRA to run all six solvers)")
        if max_runs is not None:
            print(f"DEBUG: LARGE_BENCH_MAX_RUNS active -> limiting to first {max_runs} runs")
        if gurobi_logs:
            print("Gurobi logs: enabled by default (set LARGE_BENCH_NO_GUROBI_LOGS=1 to disable)")
        else:
            print("Gurobi logs: disabled (LARGE_BENCH_NO_GUROBI_LOGS=1)")
        print("\nRunning suite...")
        print("-" * 60)

        results: List[Dict] = []
        completed_runs = 0
        run_idx = 0
        stop_early = False

        for spec in instance_specs:
            n = spec["n"]
            p = spec["p"]
            fixed = spec.get("fixed", False)
            seed_list = [None] if fixed else seeds
            for seed in seed_list:
                run_idx += 1
                case_label = f"[{run_idx}] n={n}, p={p}" + (f", seed={seed}" if seed is not None else " (fixed)")

                _write_live_progress(
                    progress_path=progress_path,
                    completed_runs=completed_runs,
                    total_runs=total_runs,
                    current_case=case_label,
                    last_completed_case=(results[-1].get("_case_label") if results else None),
                    live_csv_path=live_csv_path,
                    run_log_path=run_log_path,
                )

                print(f"  {case_label} ...", end=" ", flush=True)
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
                        use_phase1=True,
                        stage_progress=True,
                        heartbeat_sec=(0.0 if gurobi_logs else 20.0),
                        skip_f3=skip_f3,
                        include_md_and_phase12=include_extras,
                        gurobi_logs=gurobi_logs,
                    )
                    r["seed"] = seed
                    r["fixed"] = fixed
                    r["_case_label"] = case_label
                    results.append(r)
                    if include_extras:
                        print(
                            f"F3={_fmt_time(r.get('f3_time'))}s, NormB={_fmt_time(r.get('norm_time'))}s, "
                            f"NewB={_fmt_time(r.get('new_time'))}s, MD={_fmt_time(r.get('md_time'))}s, "
                            f"MDP={_fmt_time(r.get('mdp_time'))}s, P12={_fmt_time(r.get('p12_time'))}s "
                            f"(fastest: {_fastest_winner(r, include_extras, skip_f3)})"
                        )
                        print(
                            "      Obj: "
                            f"F3={_fmt_obj(r.get('f3_obj'))}  "
                            f"NormB={_fmt_obj(r.get('norm_obj'))}  "
                            f"NewB={_fmt_obj(r.get('new_obj'))}  "
                            f"MD={_fmt_obj(r.get('md_obj'))}  "
                            f"MDP={_fmt_obj(r.get('mdp_obj'))}  "
                            f"P12={_fmt_obj(r.get('p12_obj'))}  |  "
                            f"{'Match' if r.get('match') else 'MISMATCH'}"
                        )
                        _print_gap_summary(r, include_extras=include_extras, skip_f3=skip_f3)
                        _print_solver_warnings(r, include_extras=include_extras, skip_f3=skip_f3)
                    else:
                        print(
                            f"F3={_fmt_time(r.get('f3_time'))}s, NormB={_fmt_time(r.get('norm_time'))}s, "
                            f"NewB={_fmt_time(r.get('new_time'))}s "
                            f"(fastest: {_fastest_winner(r, include_extras, skip_f3)})"
                        )
                        print(
                            f"      Obj: F3={_fmt_obj(r.get('f3_obj'))}  "
                            f"NormB={_fmt_obj(r.get('norm_obj'))}  "
                            f"NewB={_fmt_obj(r.get('new_obj'))}  |  "
                            f"{'Match' if r.get('match') else 'MISMATCH'}"
                        )
                        _print_gap_summary(r, include_extras=include_extras, skip_f3=skip_f3)
                        _print_solver_warnings(r, include_extras=include_extras, skip_f3=skip_f3)
                except Exception as e:
                    print(f"ERROR: {e}")
                    err_row: Dict = {
                        "n": n,
                        "p": p,
                        "seed": seed,
                        "fixed": fixed,
                        "f3_time": None,
                        "norm_time": None,
                        "new_time": None,
                        "match": False,
                        "error": str(e),
                        "_case_label": case_label,
                    }
                    if include_extras:
                        err_row["md_time"] = err_row["mdp_time"] = err_row["p12_time"] = None
                    results.append(err_row)

                completed_runs += 1
                save_results_csv([{k: v for k, v in row.items() if not k.startswith("_")} for row in results], live_csv_path)
                _write_live_progress(
                    progress_path=progress_path,
                    completed_runs=completed_runs,
                    total_runs=total_runs,
                    current_case="IDLE_BETWEEN_CASES",
                    last_completed_case=case_label,
                    live_csv_path=live_csv_path,
                    run_log_path=run_log_path,
                )

                if max_runs is not None and completed_runs >= max_runs:
                    stop_early = True
                    print(f"\nStopping early due to LARGE_BENCH_MAX_RUNS={max_runs}.")
                    break
            if stop_early:
                break

        agg = aggregate_by_instance(results)
        print_timing_table(results, agg)
        print_detailed_results(results)

        # Summary: fastest counts
        print("\n" + "=" * 95)
        print("SUMMARY")
        print("=" * 95)
        total = len([r for r in results if "error" not in r])
        matches = sum(1 for r in results if r.get("match"))
        f3_wins = norm_wins = new_wins = 0
        md_wins = mdp_wins = p12_wins = 0
        for r in results:
            w = _fastest_winner(r, include_extras, skip_f3)
            if w is None:
                continue
            if w == "F3":
                f3_wins += 1
            elif w == "NormB":
                norm_wins += 1
            elif w == "NewB":
                new_wins += 1
            elif w == "MD":
                md_wins += 1
            elif w == "MDP":
                mdp_wins += 1
            elif w == "P12":
                p12_wins += 1

        print(f"Total runs:             {total}")
        print(f"Objectives match:       {matches}/{total}")
        if not skip_f3:
            print(f"F3 fastest:             {f3_wins} runs")
        print(f"Normal Benders fastest: {norm_wins} runs")
        print(f"New Benders fastest:    {new_wins} runs")
        if include_extras:
            print(f"MD fastest:             {md_wins} runs")
            print(f"MD Pareto fastest:      {mdp_wins} runs")
            print(f"Pareto P12 fastest:     {p12_wins} runs")

        csv_path = os.path.join(_this_dir, "large_instance_results.csv")
        save_results_csv([{k: v for k, v in row.items() if not k.startswith("_")} for row in results], csv_path)

        ok = matches == total
        print("\n" + ("All objectives matched." if ok else "WARNING: Some objectives did not match."))
        return ok


if __name__ == "__main__":
    ok = main()
