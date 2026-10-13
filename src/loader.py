"""
loader.py — read CVRPLIB `.vrp` files and generate synthetic instances.

CVRPLIB format (Augerat et al.) looks like:

    NAME : A-n32-k5
    COMMENT : (Augerat et al, No of trucks: 5, Optimal value: 784)
    TYPE : CVRP
    DIMENSION : 32
    EDGE_WEIGHT_TYPE : EUC_2D
    CAPACITY : 100
    NODE_COORD_SECTION
     1 82 76
     2 96 44
     ...
    DEMAND_SECTION
    1 0
    2 19
    ...
    DEPOT_SECTION
     1
    -1
    EOF

Nodes are 1-based in the file; we re-index so the depot is 0 and customers
are 1..n, which is what common.py expects. Distances are Euclidean rounded
to the nearest integer (the EUC_2D convention the published optima use —
if you don't round, your MILP will "beat" the official optimum and that is
a bug, not a discovery).

Usage:
    python src/loader.py data/instances/P-n16-k8.vrp
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import Instance  # noqa: E402


def euclidean_matrix(coords: np.ndarray, rounded: bool = True) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    return np.rint(d).astype(int) if rounded else d


def load_cvrplib(path: str | Path) -> Instance:
    text = Path(path).read_text()
    header, body = {}, text

    # --- header lines are "KEY : value" until the first *_SECTION
    for line in text.splitlines():
        if "_SECTION" in line:
            break
        if ":" in line:
            k, v = line.split(":", 1)
            header[k.strip().upper()] = v.strip()

    name = header.get("NAME", Path(path).stem)
    dim = int(header["DIMENSION"])
    capacity = int(header["CAPACITY"])
    if header.get("EDGE_WEIGHT_TYPE", "EUC_2D") != "EUC_2D":
        raise ValueError(f"{name}: only EUC_2D instances are supported")

    # --- number of trucks: from name "...-kK" or from the comment
    m = re.search(r"-k(\d+)", name) or re.search(r"trucks:\s*(\d+)", header.get("COMMENT", ""))
    n_vehicles = int(m.group(1)) if m else dim  # fall back: unlimited fleet

    # --- published optimum, if the comment carries one
    m = re.search(r"Optimal value:\s*(\d+)", header.get("COMMENT", ""))
    optimum = int(m.group(1)) if m else None

    def section(tag: str) -> list[str]:
        start = body.index(tag) + len(tag)
        end = min([i for i in (body.find(s, start) for s in
                   ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF"))
                   if i != -1] or [len(body)])
        return [l for l in body[start:end].strip().splitlines() if l.strip()]

    coords = np.zeros((dim, 2))
    for line in section("NODE_COORD_SECTION"):
        i, x, y = line.split()[:3]
        coords[int(i) - 1] = (float(x), float(y))

    demand = np.zeros(dim, dtype=int)
    for line in section("DEMAND_SECTION"):
        i, q = line.split()[:2]
        demand[int(i) - 1] = int(q)

    depot_idx = int(section("DEPOT_SECTION")[0]) - 1
    if depot_idx != 0:  # move depot to row 0 so common.py's convention holds
        order = [depot_idx] + [i for i in range(dim) if i != depot_idx]
        coords, demand = coords[order], demand[order]

    return Instance(
        name=name, coords=coords, demand=demand, capacity=capacity,
        n_vehicles=n_vehicles, dist=euclidean_matrix(coords), optimum=optimum,
    )


def generate_synthetic(n_customers: int, seed: int = 0, capacity: int = 100,
                       area: int = 100, demand_range=(1, 25)) -> Instance:
    """
    Synthetic instance for the scalability study.
    Realism justification (put this in the report): customers uniformly
    spread over a square service area with the depot at the centre, integer
    demands in a bounded range, and a fleet sized at ceil(1.2 * total/Q) —
    the same structure Augerat's benchmark instances follow.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, area, size=(n_customers + 1, 2))
    coords[0] = (area / 2, area / 2)
    demand = rng.integers(demand_range[0], demand_range[1] + 1, size=n_customers + 1)
    demand[0] = 0
    k = int(np.ceil(1.2 * demand.sum() / capacity))
    return Instance(
        name=f"synth-n{n_customers}-k{k}-s{seed}", coords=coords, demand=demand,
        capacity=capacity, n_vehicles=k, dist=euclidean_matrix(coords), optimum=None,
    )


if __name__ == "__main__":
    inst = load_cvrplib(sys.argv[1] if len(sys.argv) > 1 else "data/instances/P-n16-k8.vrp")
    print(f"{inst.name}: {inst.n_customers} customers, K={inst.n_vehicles}, "
          f"Q={inst.capacity}, total demand={inst.demand.sum()}, optimum={inst.optimum}")
    print("depot at", inst.coords[0], "| first customer demand", inst.demand[1])
    print("dist[0,1] =", inst.dist[0, 1], "| matrix shape", inst.dist.shape)
