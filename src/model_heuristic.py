"""
model_heuristic.py - greedy + simulated annealing for the CVRP.

greedy (nearest neighbour) gives the first routes, then SA improves them
with relocate / swap / 2-opt moves. CVRP is NP-hard so MILP gets slow,
this gives a good answer fast (no proof of optimal though).

routes = list of customer ids, depot 0 not written (same as common.py).
run: python src/model_heuristic.py data/instances/P-n16-k8.vrp
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from common import Instance, Solution  # noqa: E402


# ---- stage 1: greedy ----

def greedy(inst: Instance) -> list[list[int]]:
    """nearest customer that still fits, else back to depot and new truck"""
    unvisited = set(inst.customers)
    routes = []
    while unvisited:
        route, load, here = [], 0, 0          # new truck, empty, at depot
        while True:
            fits = [c for c in unvisited if load + inst.demand[c] <= inst.capacity]
            if not fits:
                break                         # full, go home
            # closest one (tie -> smaller id)
            nxt = min(fits, key=lambda c: (inst.dist[here, c], c))
            route.append(nxt)
            load += inst.demand[nxt]
            unvisited.remove(nxt)
            here = nxt
        routes.append(route)
    return routes


# ---- stage 2: moves ----
# each move returns NEW routes (input not changed), None if a truck overloads

def relocate(inst: Instance, routes, r1: int, i: int, r2: int, j: int):
    """move customer i of route r1 to position j of route r2"""
    new = [list(r) for r in routes]
    customer = new[r1].pop(i)
    new[r2].insert(j, customer)
    # only r2 got heavier so only check r2
    if r1 != r2 and common.route_load(inst, new[r2]) > inst.capacity:
        return None
    return [r for r in new if r]              # drop empty route (1 truck less)


def swap(inst: Instance, routes, r1: int, i: int, r2: int, j: int):
    """swap 2 customers on different routes"""
    if r1 == r2:
        return None
    new = [list(r) for r in routes]
    new[r1][i], new[r2][j] = new[r2][j], new[r1][i]
    # both loads change -> check both
    if (common.route_load(inst, new[r1]) > inst.capacity
            or common.route_load(inst, new[r2]) > inst.capacity):
        return None
    return new


def two_opt(inst: Instance, routes, r: int, i: int, j: int):
    """reverse route r from i to j, removes a crossing. same load so no check"""
    new = [list(rt) for rt in routes]
    new[r][i:j + 1] = reversed(new[r][i:j + 1])
    return new


def random_move(inst: Instance, routes, rng: random.Random):
    """pick a random move with random positions"""
    move = rng.choice(("relocate", "swap", "two_opt"))
    r1 = rng.randrange(len(routes))
    i = rng.randrange(len(routes[r1]))

    if move == "relocate":
        r2 = rng.randrange(len(routes))
        # r1 is 1 shorter after pop
        size = len(routes[r2]) - 1 if r2 == r1 else len(routes[r2])
        j = rng.randint(0, size)
        if r1 == r2 and i == j:
            return None                       # same spot, nothing changes
        return relocate(inst, routes, r1, i, r2, j)

    if move == "swap":
        if len(routes) < 2:
            return None
        r2 = rng.randrange(len(routes))
        j = rng.randrange(len(routes[r2]))
        return swap(inst, routes, r1, i, r2, j)

    # 2-opt needs 2+ customers
    if len(routes[r1]) < 2:
        return None
    a, b = sorted(rng.sample(range(len(routes[r1])), 2))
    return two_opt(inst, routes, r1, a, b)


# ---- stage 3: simulated annealing ----

def objective(inst: Instance, routes, penalty: float) -> float:
    """distance + penalty if more routes than trucks (K).
    penalty = load left on the lightest extra route, so moving customers off
    it lowers the penalty step by step. 0 when routes <= K"""
    extra_routes = len(routes) - inst.n_vehicles
    if extra_routes <= 0:
        return common.solution_cost(inst, routes)
    loads = sorted(common.route_load(inst, r) for r in routes)
    return common.solution_cost(inst, routes) + penalty * sum(loads[:extra_routes])


def initial_temperature(inst: Instance, routes, penalty: float, rng: random.Random,
                        samples: int = 100) -> float:
    """T0 = avg |delta| of 100 random moves (avg bad move accepted ~37% at start)"""
    start = objective(inst, routes, penalty)
    deltas = []
    while len(deltas) < samples:
        new = random_move(inst, routes, rng)
        if new is not None:                   # skip overloaded ones
            deltas.append(abs(objective(inst, new, penalty) - start))
    return sum(deltas) / len(deltas)


def solve(inst: Instance, time_limit: float = 300.0, seed: int = 0, T0: float | None = None,
          alpha: float = 0.995, iters_per_temp: int | None = None,
          T_min: float = 0.01, **kw) -> Solution:
    """SA from the greedy routes.
    better move -> always take. worse by delta -> take with prob exp(-delta/T)
    (this is how it gets out of local optima). T *= alpha every iters_per_temp
    moves. stop when T < T_min or time is up. return best, not current"""
    start_time = time.perf_counter()
    rng = random.Random(seed)                 # same seed = same run
    penalty = float(inst.dist.max())

    start_routes = greedy(inst)
    current, current_cost = start_routes, objective(inst, start_routes, penalty)
    # best only counts plans with <= K trucks
    best, best_cost = None, math.inf
    if len(current) <= inst.n_vehicles:
        best, best_cost = current, current_cost

    T = T0 if T0 is not None else initial_temperature(inst, current, penalty, rng)
    start_T = T
    if iters_per_temp is None:
        iters_per_temp = 10 * inst.n_customers  # bigger problem, more tries per T

    history = []                              # best cost each iteration (for the plot)
    iterations = 0
    while T > T_min and time.perf_counter() - start_time < time_limit:
        for _ in range(iters_per_temp):
            iterations += 1
            new = random_move(inst, current, rng)
            if new is not None:
                new_cost = objective(inst, new, penalty)
                delta = new_cost - current_cost
                # accept rule
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    current, current_cost = new, new_cost
                    if len(current) <= inst.n_vehicles and current_cost < best_cost:
                        best, best_cost = current, current_cost
            history.append(best_cost)
        T *= alpha                            # cool down

    assert best is not None, "SA never found a plan that fits in K trucks"
    ok, why = common.is_feasible(inst, best)
    assert ok, f"SA returned an infeasible solution: {why}"
    return Solution(
        routes=best,
        cost=common.solution_cost(inst, best),
        runtime=time.perf_counter() - start_time,
        method="sa",
        status="feasible",
        extra={"seed": seed, "T0": start_T, "alpha": alpha, "iters_per_temp": iters_per_temp,
               "iterations": iterations, "greedy_cost": common.solution_cost(inst, start_routes),
               "history": history},
    )


if __name__ == "__main__":
    from loader import load_cvrplib

    path = sys.argv[1] if len(sys.argv) > 1 else "data/instances/P-n16-k8.vrp"
    inst = load_cvrplib(path)

    routes = greedy(inst)
    print("Greedy:", common.solution_cost(inst, routes), "|", common.is_feasible(inst, routes))

    sol = solve(inst, seed=0)
    print(common.describe(inst, sol))
    print(f"T0={sol.extra['T0']:.1f}  iterations={sol.extra['iterations']:,}  "
          f"greedy={sol.extra['greedy_cost']:.0f}")
