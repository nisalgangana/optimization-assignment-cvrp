"""
evaluate.py - runs MILP and SA on all instances, saves results/results.csv

one row per run. cost is recomputed here with common.solution_cost so both
methods are scored the same way. synthetic n=10,20,40,80 = scalability study
(where does the MILP break?).

    python src/evaluate.py --quick     # only P-n16-k8 -> results/results_quick.csv
    python src/evaluate.py --full      # everything (MILP takes long)
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import loader  # noqa: E402
import model_heuristic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INSTANCE_DIR = ROOT / "data" / "instances"
RESULTS_CSV = ROOT / "results" / "results.csv"
QUICK_CSV = ROOT / "results" / "results_quick.csv"   # quick test goes here, not over the real one
COLUMNS = ["instance", "n", "method", "seed", "cost", "optimum", "gap_pct",
           "runtime_s", "status", "bound"]


def load_milp():
    """Nisal's model_milp.solve, or None if not there"""
    try:
        import model_milp
        return model_milp.solve
    except (ImportError, AttributeError):
        print("MILP not ready yet (model_milp.solve missing) - running SA only.\n")
        return None


def make_row(inst, sol, seed):
    """one csv row. no routes (MILP found nothing) -> keep cost inf, not 0"""
    cost = common.solution_cost(inst, sol.routes) if sol.routes else sol.cost
    gap = sol.gap_to(inst.optimum) if sol.routes else None
    return {
        "instance": inst.name, "n": inst.n_customers, "method": sol.method,
        "seed": seed, "cost": cost, "optimum": inst.optimum,
        "gap_pct": None if gap is None else round(gap, 2),
        "runtime_s": round(sol.runtime, 3), "status": sol.status, "bound": sol.bound,
    }


def run(quick: bool):
    milp_solve = load_milp()
    seeds = [0] if quick else [0, 1, 2, 3, 4]
    sa_limit, milp_limit = (10, 60) if quick else (60, 600)

    # cvrplib ones (known optimum)
    paths = [INSTANCE_DIR / "P-n16-k8.vrp"] if quick else sorted(INSTANCE_DIR.glob("*.vrp"))
    benchmarks = sorted((loader.load_cvrplib(p) for p in paths), key=lambda i: i.n_customers)
    # random ones, growing size
    synthetic = [] if quick else [loader.generate_synthetic(n, seed=0) for n in (10, 20, 40, 80)]

    rows = []
    for inst in benchmarks + synthetic:
        if milp_solve is not None:
            sol = milp_solve(inst, time_limit=milp_limit)   # milp once
            rows.append(make_row(inst, sol, seed=None))
            print(f"{inst.name:18s} milp  cost={rows[-1]['cost']:8.0f}  {sol.status}  "
                  f"{sol.runtime:7.1f}s", flush=True)
        for seed in seeds:                                  # sa 5 times (random)
            sol = model_heuristic.solve(inst, time_limit=sa_limit, seed=seed)
            rows.append(make_row(inst, sol, seed))
            print(f"{inst.name:18s} sa    cost={rows[-1]['cost']:8.0f}  seed={seed}  "
                  f"{sol.runtime:7.1f}s", flush=True)
    return rows


def save(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {path.relative_to(ROOT)}")


def print_summary(rows):
    """mean +- std per instance and method"""
    groups = {}
    for r in rows:
        groups.setdefault((r["n"], r["instance"], r["method"]), []).append(r)
    print(f"\n{'instance':18s} {'method':6s} {'runs':>4s} {'cost mean +- std':>18s} "
          f"{'gap %':>7s} {'time s':>8s}")
    for (_, name, method), group in sorted(groups.items()):
        costs = [r["cost"] for r in group]
        std = statistics.stdev(costs) if len(costs) > 1 else 0.0
        gap = "-" if group[0]["gap_pct"] is None else f"{statistics.mean(r['gap_pct'] for r in group):.2f}"
        time_s = statistics.mean(r["runtime_s"] for r in group)
        print(f"{name:18s} {method:6s} {len(group):4d} {statistics.mean(costs):10.1f} +- {std:5.1f} "
              f"{gap:>7s} {time_s:8.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MILP and SA on all instances.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="P-n16-k8 only, short time limits")
    mode.add_argument("--full", action="store_true", help="all instances, full time limits")
    args = parser.parse_args()

    results = run(quick=args.quick)
    save(results, QUICK_CSV if args.quick else RESULTS_CSV)
    print_summary(results)
