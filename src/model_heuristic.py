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


# ---------------------------------------------------------------------------
# Stage 2 — neighbourhood moves
# Each move is a small pure function: it never changes the routes it is given,
# it returns a NEW list of routes, or None when the move would overload a truck.
# ---------------------------------------------------------------------------

def relocate(inst: Instance, routes, r1: int, i: int, r2: int, j: int):
    """Take the customer at position i of route r1 and insert it at position j
    of route r2 (r2 may be the same route). Only the receiving route gains
    load, so only that route needs a capacity check."""
    new = [list(r) for r in routes]
    customer = new[r1].pop(i)
    new[r2].insert(j, customer)
    if r1 != r2 and common.route_load(inst, new[r2]) > inst.capacity:
        return None
    return [r for r in new if r]              # a route that became empty is closed


def swap(inst: Instance, routes, r1: int, i: int, r2: int, j: int):
    """Exchange customer i of route r1 with customer j of route r2 (two
    different routes). Both routes change load, so both are checked."""
    if r1 == r2:
        return None
    new = [list(r) for r in routes]
    new[r1][i], new[r2][j] = new[r2][j], new[r1][i]
    if (common.route_load(inst, new[r1]) > inst.capacity
            or common.route_load(inst, new[r2]) > inst.capacity):
        return None
    return new


def two_opt(inst: Instance, routes, r: int, i: int, j: int):
    """Reverse the part of route r between positions i and j (i < j). This
    removes a place where the route crosses itself. The same customers stay on
    the route, so the load does not change and it is always feasible."""
    new = [list(rt) for rt in routes]
    new[r][i:j + 1] = reversed(new[r][i:j + 1])
    return new


def random_move(inst: Instance, routes, rng: random.Random):
    """Pick one of the three moves at random, with random positions.
    Returns the new routes, or None if the draw was infeasible or did nothing."""
    move = rng.choice(("relocate", "swap", "two_opt"))
    r1 = rng.randrange(len(routes))
    i = rng.randrange(len(routes[r1]))

    if move == "relocate":
        r2 = rng.randrange(len(routes))
        # after removing the customer, route r1 is one shorter
        size = len(routes[r2]) - 1 if r2 == r1 else len(routes[r2])
        j = rng.randint(0, size)
        if r1 == r2 and i == j:
            return None                       # put back in the same place
        return relocate(inst, routes, r1, i, r2, j)

    if move == "swap":
        if len(routes) < 2:
            return None
        r2 = rng.randrange(len(routes))
        j = rng.randrange(len(routes[r2]))
        return swap(inst, routes, r1, i, r2, j)

    # two_opt needs at least two customers on the route
    if len(routes[r1]) < 2:
        return None
    a, b = sorted(rng.sample(range(len(routes[r1])), 2))
    return two_opt(inst, routes, r1, a, b)


if __name__ == "__main__":
    from loader import load_cvrplib

    path = sys.argv[1] if len(sys.argv) > 1 else "data/instances/P-n16-k8.vrp"
    inst = load_cvrplib(path)

    routes = greedy(inst)
    print("Greedy:", common.solution_cost(inst, routes), "|", common.is_feasible(inst, routes))
    for k, route in enumerate(routes, 1):
        print(f"  route {k}: {route}  load {common.route_load(inst, route)}/{inst.capacity}")
