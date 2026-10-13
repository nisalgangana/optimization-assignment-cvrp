"""
common.py — shared data structures for every solver in this project.

Both the exact MILP (model_milp.py) and the heuristic (model_heuristic.py)
read an `Instance` and return a `Solution`. Keeping the cost and feasibility
checks HERE (and nowhere else) guarantees both methods are compared on
exactly the same numbers.

Node convention used everywhere in this repo:
    node 0           = depot
    nodes 1 .. n     = customers
A solution is a list of routes; each route is a list of customer ids
(depot NOT included — it is implied at the start and end of every route).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Instance:
    name: str
    coords: np.ndarray          # shape (n+1, 2); row 0 is the depot
    demand: np.ndarray          # shape (n+1,);   demand[0] == 0
    capacity: int               # vehicle capacity Q
    n_vehicles: int             # fleet size K (upper bound on routes used)
    dist: np.ndarray            # shape (n+1, n+1) integer distance matrix
    optimum: Optional[int] = None   # published optimal cost, if known

    @property
    def n_customers(self) -> int:
        return len(self.demand) - 1

    @property
    def customers(self) -> range:
        return range(1, self.n_customers + 1)


@dataclass
class Solution:
    routes: List[List[int]]     # e.g. [[3, 7, 1], [5, 2], ...]
    cost: float                 # total distance, always from solution_cost()
    runtime: float              # wall-clock seconds the solver took
    method: str                 # "milp", "greedy", "sa", ...
    status: str                 # "optimal" | "feasible" | "timeout" | "infeasible"
    bound: Optional[float] = None   # best lower bound (MILP only)
    extra: dict = field(default_factory=dict)  # anything else (seed, iterations...)

    def gap_to(self, reference: Optional[float]) -> Optional[float]:
        """Percentage gap of this solution above a reference cost."""
        if reference is None or reference == 0:
            return None
        return 100.0 * (self.cost - reference) / reference


# ---------------------------------------------------------------------------
# The two functions every solver must rely on
# ---------------------------------------------------------------------------

def route_cost(inst: Instance, route: List[int]) -> float:
    """Distance of depot -> route[0] -> ... -> route[-1] -> depot."""
    if not route:
        return 0.0
    total = inst.dist[0, route[0]] + inst.dist[route[-1], 0]
    for a, b in zip(route[:-1], route[1:]):
        total += inst.dist[a, b]
    return float(total)


def solution_cost(inst: Instance, routes: List[List[int]]) -> float:
    return sum(route_cost(inst, r) for r in routes)


def route_load(inst: Instance, route: List[int]) -> int:
    return int(sum(inst.demand[c] for c in route))


def is_feasible(inst: Instance, routes: List[List[int]]) -> tuple[bool, str]:
    """
    A solution is feasible when:
      1. every customer is visited exactly once,
      2. no route exceeds vehicle capacity,
      3. no more than K routes are used.
    Returns (ok, reason) so callers can print why something failed.
    """
    visited = [c for r in routes for c in r]
    if sorted(visited) != list(inst.customers):
        missing = set(inst.customers) - set(visited)
        dupes = {c for c in visited if visited.count(c) > 1}
        return False, f"customer coverage wrong (missing={missing}, duplicated={dupes})"
    for i, r in enumerate(routes):
        if route_load(inst, r) > inst.capacity:
            return False, f"route {i} load {route_load(inst, r)} > capacity {inst.capacity}"
    if len([r for r in routes if r]) > inst.n_vehicles:
        return False, f"{len(routes)} routes used but only {inst.n_vehicles} vehicles"
    return True, "ok"


def describe(inst: Instance, sol: Solution) -> str:
    ok, why = is_feasible(inst, sol.routes)
    lines = [
        f"{inst.name}  method={sol.method}  status={sol.status}",
        f"  cost={sol.cost:.0f}  optimum={inst.optimum}  "
        f"gap={sol.gap_to(inst.optimum):.2f}%" if inst.optimum else f"  cost={sol.cost:.0f}",
        f"  runtime={sol.runtime:.2f}s  feasible={ok} ({why})",
    ]
    for i, r in enumerate(sol.routes):
        lines.append(f"  route {i}: 0 -> {' -> '.join(map(str, r))} -> 0  "
                     f"(load {route_load(inst, r)}/{inst.capacity}, dist {route_cost(inst, r):.0f})")
    return "\n".join(lines)
