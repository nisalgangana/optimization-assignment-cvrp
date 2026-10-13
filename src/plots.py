"""
plots.py - all figures for the results section (from results/results.csv)

    cost_comparison.png    optimum vs MILP vs SA cost per instance
    runtime_vs_n.png       runtime vs n (log scale)
    gap_vs_n.png           gap to optimum vs n (SA mean +- std)
    sa_convergence.png     SA best cost per iteration, A-n32-k5 seed 0
    routes_<name>.png      route map, 1 colour per truck

each function returns the fig so notebooks can show it.
plot_routes(inst, sol) works for MILP and SA solutions.

    python src/plots.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import loader  # noqa: E402
import model_heuristic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results" / "results.csv"
FIGURES = ROOT / "results" / "figures"
COLOURS = {"sa": "#2a78d6", "milp": "#eb6834", "optimum": "#555555"}
LABELS = {"sa": "Simulated annealing (mean of seeds)", "milp": "MILP (exact)",
          "optimum": "Proven optimum"}
ROUTE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#4a3aa7", "#e34948", "#7a7a7a", "#00a3b4"]


def read_results(path=RESULTS_CSV):
    with open(path) as f:
        return list(csv.DictReader(f))


def group_by_instance(rows, method):
    """{n: rows} for one method, sorted by n"""
    groups = {}
    for r in rows:
        if r["method"] == method:
            groups.setdefault(int(r["n"]), []).append(r)
    return dict(sorted(groups.items()))


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    print(f"saved results/figures/{name}")
    return fig


def plot_routes(inst, sol, filename=None):
    """route map: depot, customers, each truck own colour (MILP or SA sol)"""
    fig, ax = plt.subplots(figsize=(7, 7))
    depot = inst.coords[0]
    for k, route in enumerate(sol.routes):
        path = [depot] + [inst.coords[c] for c in route] + [depot]
        xs, ys = zip(*path)
        load = common.route_load(inst, route)
        ax.plot(xs, ys, color=ROUTE_COLOURS[k % len(ROUTE_COLOURS)], linewidth=2,
                label=f"Route {k + 1} (load {load}/{inst.capacity})")
    ax.scatter(inst.coords[1:, 0], inst.coords[1:, 1], color="black", s=20, zorder=3)
    ax.scatter(*depot, color="red", marker="s", s=100, zorder=4, label="Depot")
    ax.set_title(f"{inst.name}: {sol.method.upper()} routes, cost {sol.cost:.0f}")
    ax.set_aspect("equal")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return save(fig, filename or f"routes_{inst.name}_{sol.method}.png")


def cost_comparison(rows):
    """bars per instance: optimum, MILP cost, SA mean cost"""
    names, optimum, milp, sa = [], [], [], []
    for n, group in group_by_instance(rows, "sa").items():
        if not group[0]["optimum"]:
            continue                                   # synthetic has no optimum, skip
        name = group[0]["instance"]
        names.append(name)
        optimum.append(float(group[0]["optimum"]))
        sa.append(statistics.mean(float(r["cost"]) for r in group))
        m = [r for r in rows if r["method"] == "milp" and r["instance"] == name]
        milp.append(float(m[0]["cost"]) if m and m[0]["cost"] != "inf" else 0.0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = range(len(names))
    width = 0.27
    ax.bar([x - width for x in xs], optimum, width, color=COLOURS["optimum"], label=LABELS["optimum"])
    if any(milp):
        ax.bar(list(xs), milp, width, color=COLOURS["milp"], label=LABELS["milp"])
        for x, c in zip(xs, milp):
            if c == 0:                                 # milp found nothing in time
                ax.text(x, 50, "MILP: no\nsolution\nin 600 s", ha="center", fontsize=8,
                        color=COLOURS["milp"])
    ax.bar([x + width for x in xs], sa, width, color=COLOURS["sa"], label=LABELS["sa"])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names)
    ax.set_ylabel("Total distance")
    ax.set_title("Cost on the benchmark instances (lower is better)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    return save(fig, "cost_comparison.png")


def runtime_vs_n(rows):
    """mean runtime vs n, log y, 1 line per method"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in ("milp", "sa"):
        groups = group_by_instance(rows, method)
        if not groups:
            continue
        ns = list(groups)
        times = [statistics.mean(float(r["runtime_s"]) for r in g) for g in groups.values()]
        ax.plot(ns, times, marker="o", color=COLOURS[method], label=LABELS[method])
    ax.set_yscale("log")
    ax.set_xlabel("Number of customers n")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.set_title("Runtime against problem size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save(fig, "runtime_vs_n.png")


def gap_vs_n(rows):
    """gap to optimum vs n, SA with std error bars"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in ("milp", "sa"):
        groups = {n: g for n, g in group_by_instance(rows, method).items() if g[0]["gap_pct"]}
        if not groups:
            continue
        ns = list(groups)
        gaps = [[float(r["gap_pct"]) for r in g] for g in groups.values()]
        means = [statistics.mean(g) for g in gaps]
        stds = [statistics.stdev(g) if len(g) > 1 else 0.0 for g in gaps]
        ax.errorbar(ns, means, yerr=stds, marker="o", capsize=4,
                    color=COLOURS[method], label=LABELS[method])
    ax.set_xlabel("Number of customers n")
    ax.set_ylabel("Gap to optimum (%)")
    ax.set_title("Solution quality against problem size")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save(fig, "gap_vs_n.png")


def sa_convergence(inst, sol):
    """best cost per iteration + optimum line"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sol.extra["history"], color=COLOURS["sa"], label="SA best cost so far")
    if inst.optimum:
        ax.axhline(inst.optimum, color="black", linewidth=1, linestyle="--",
                   label=f"Optimum ({inst.optimum})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total distance")
    ax.set_title(f"SA convergence on {inst.name} (seed {sol.extra['seed']})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save(fig, "sa_convergence.png")


if __name__ == "__main__":
    rows = read_results()
    cost_comparison(rows)
    runtime_vs_n(rows)
    gap_vs_n(rows)

    # 1 SA run -> convergence plot + route map
    inst = loader.load_cvrplib(ROOT / "data" / "instances" / "A-n32-k5.vrp")
    sol = model_heuristic.solve(inst, seed=0)
    sa_convergence(inst, sol)
    plot_routes(inst, sol)
