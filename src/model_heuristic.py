"""
model_heuristic.py — greedy construction + Simulated Annealing for the CVRP.

WHAT: the heuristic side of the assignment. A greedy nearest-neighbour rule
builds a first feasible set of routes, then Simulated Annealing (SA) keeps
changing them with small moves (relocate, swap, 2-opt) to shorten the total
distance.
WHY: the CVRP is NP-hard, so the exact MILP becomes too slow as the number of
customers grows. A heuristic gives no proof of optimality, but it finds a good
plan in a few seconds even for large instances.

Conventions (from common.py): node 0 is the depot, a route is a list of
customer ids such as [3, 7, 1] with the depot implied at both ends, and every
cost is computed with common.solution_cost.

Usage:
    python src/model_heuristic.py data/instances/P-n16-k8.vrp
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


# ---------------------------------------------------------------------------
# Stage 1 — greedy construction
# ---------------------------------------------------------------------------

def greedy(inst: Instance) -> list[list[int]]:
    """Capacity-aware nearest neighbour.

    Start at the depot and keep driving to the nearest customer that has not
    been visited yet and still fits in the truck. When no customer fits, the
    truck goes back to the depot and a new route starts. It is fast and always
    feasible on capacity, but short-sighted: the last routes have to collect
    whatever customers are left, wherever they are.
    """
    unvisited = set(inst.customers)
    routes = []
    while unvisited:
        route, load, here = [], 0, 0          # every route starts empty at the depot
        while True:
            fits = [c for c in unvisited if load + inst.demand[c] <= inst.capacity]
            if not fits:
                break                         # truck is full: go back to the depot
            # nearest customer that fits (ties broken by the smaller id)
            nxt = min(fits, key=lambda c: (inst.dist[here, c], c))
            route.append(nxt)
            load += inst.demand[nxt]
            unvisited.remove(nxt)
            here = nxt
        routes.append(route)
    return routes


if __name__ == "__main__":
    from loader import load_cvrplib

    path = sys.argv[1] if len(sys.argv) > 1 else "data/instances/P-n16-k8.vrp"
    inst = load_cvrplib(path)

    routes = greedy(inst)
    print("Greedy:", common.solution_cost(inst, routes), "|", common.is_feasible(inst, routes))
    for k, route in enumerate(routes, 1):
        print(f"  route {k}: {route}  load {common.route_load(inst, route)}/{inst.capacity}")
