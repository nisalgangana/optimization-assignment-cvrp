"""
tuning.py - small SA parameter test on A-n32-k5 (opt 784)

change one thing at a time, 5 seeds each:
    alpha: 0.95, 0.99, 0.995 (default), 0.999
    T0: 0.5x, 1x, 2x the auto value
saves results/tuning.csv + results/figures/tuning.png
used to justify alpha=0.995 and auto T0 in the report.

    python src/tuning.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import loader  # noqa: E402
import model_heuristic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INSTANCE = ROOT / "data" / "instances" / "A-n32-k5.vrp"
SEEDS = [0, 1, 2, 3, 4]
ALPHAS = [0.95, 0.99, 0.995, 0.999]
T0_FACTORS = [0.5, 1.0, 2.0]


def run_setting(inst, parameter, value, **kw):
    """run SA for every seed with these settings"""
    rows = []
    for seed in SEEDS:
        sol = model_heuristic.solve(inst, seed=seed, **kw)
        rows.append({"parameter": parameter, "value": value, "seed": seed, "cost": sol.cost,
                     "gap_pct": round(sol.gap_to(inst.optimum), 2),
                     "runtime_s": round(sol.runtime, 2)})
    costs = [r["cost"] for r in rows]
    print(f"{parameter:6s} {value:>6}  mean cost {statistics.mean(costs):7.1f}  "
          f"best {min(costs):5.0f}  time {statistics.mean(r['runtime_s'] for r in rows):5.1f}s", flush=True)
    return rows


def main():
    inst = loader.load_cvrplib(INSTANCE)
    rows = []

    # 1. alpha only, rest default
    for alpha in ALPHAS:
        rows += run_setting(inst, "alpha", alpha, alpha=alpha)

    # 2. T0 only. fixed T0 for all seeds = factor x auto T0 of seed 0
    #    (so 1.0x is close to the alpha 0.995 row but not exactly same)
    auto_T0 = model_heuristic.solve(inst, seed=0).extra["T0"]
    for factor in T0_FACTORS:
        rows += run_setting(inst, "T0", factor, T0=factor * auto_T0)

    out = ROOT / "results" / "tuning.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out.relative_to(ROOT)}")
    plot(rows, auto_T0)


def plot(rows, auto_T0):
    """2 panels: gap mean +- std for each alpha / each T0"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, parameter, values, label in [
            (axes[0], "alpha", ALPHAS, "cooling rate alpha"),
            (axes[1], "T0", T0_FACTORS, f"T0 as a multiple of the automatic value ({auto_T0:.0f})")]:
        gaps = [[r["gap_pct"] for r in rows if r["parameter"] == parameter and r["value"] == v]
                for v in values]
        times = [statistics.mean(r["runtime_s"] for r in rows
                                 if r["parameter"] == parameter and r["value"] == v) for v in values]
        xs = range(len(values))
        ax.errorbar(xs, [statistics.mean(g) for g in gaps], yerr=[statistics.stdev(g) for g in gaps],
                    marker="o", capsize=4, color="#2a78d6")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{v}\n({t:.1f} s)" for v, t in zip(values, times)])
        ax.set_xlabel(label + "   (mean runtime in brackets)")
        ax.set_ylabel("Gap to optimum (%)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    fig.suptitle("SA sensitivity on A-n32-k5 (5 seeds per setting, mean ± std)")
    fig.tight_layout()
    path = ROOT / "results" / "figures" / "tuning.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
