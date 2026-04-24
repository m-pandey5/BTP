"""
New Pareto test suite wrapper.

Reuses all tests from test_pareto_benders.py, but swaps the Pareto solver
implementation to the new variant (LP-blended core enabled by default).

Run:
    python -m pytest p-hub-arc/test_pareto_benders_new.py -v
"""

from pareto_benders_hub_arc_new import solve_benders_pareto_hub_arc_new
import test_pareto_benders as base


# Force new solver for all reused tests in base module.
base.solve_benders_pareto_hub_arc = solve_benders_pareto_hub_arc_new


# Re-export all test_* names so pytest collects them from this file.
for _name in dir(base):
    if _name.startswith("test_"):
        globals()[_name] = getattr(base, _name)


def main():
    return base.main()


if __name__ == "__main__":
    main()

