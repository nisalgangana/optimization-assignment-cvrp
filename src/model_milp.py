"""
exact CVRP model (2-index formulation with MTZ constraints),
solved with CBC through PuLP.

x[i,j] = 1 if a vehicle drives directly from node i to node j.
u[i]   = load on the vehicle after it visits customer i.
The objective is the total distance travelled.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pulp

sys.path.insert(0, str(Path(__file__).parent))
import common #shared functions
from common import Instance, Solution #input,output data structures
from loader import load_cvrplib #functions for loading CVRP instances

def build_model(inst: Instance):
    """variables and objective (constraints added below)."""
    nodes = range(inst.n_customers + 1) #0 is depot. 1..n customers
    customers = list(inst.customers)
    arcs = [(i,j) for i in nodes for j in nodes if i != j]

    prob = pulp.LpProblem(f"CVRP_MTZ_{inst.name}", pulp.LpMinimize) #create empty minimization problem

    #x[i,j]: binary variable: 1 if vehicle drives from i to j
    x = prob.add_variable_dict("x_%s_%s", arcs, cat=pulp.LpBinary) #add binary variables for each arc
    # u[i]: continuous load after customer i is visited (0 at depot, <= capacity at customers)
    u = prob.add_variable_dict("u_%s", customers, cat=pulp.LpContinuous) #add continuous variables for each customer

    #total distance travelled (objective)
    prob += pulp.lpSum(int(inst.dist[i,j]) * x[i,j] for (i,j) in arcs)

    Q = inst.capacity          # vehicle capacity
    K = inst.n_vehicles        # fleet size
    q = inst.demand            # q[i] = demand of node i

    # C1: customer i has exactly one incoming arc (from depot or another customer)
    for j in customers:
        prob += pulp.lpSum(x[i, j] for i in nodes if i != j) == 1, f"C1_in_{j}"

    # C2: customer i has exactly one outgoing arc (to depot or another customer)
    for i in customers:
        prob += pulp.lpSum(x[i, j] for j in nodes if j != i) == 1, f"C2_out_{i}"

    # C3: at most K vehicles leave the depot, and all of them return
    leave = pulp.lpSum(x[0, j] for j in customers)
    enter = pulp.lpSum(x[i, 0] for i in customers)
    prob += leave <= K, "C3_fleet"
    prob += enter == leave, "C3_balance"

    # C4 (MTZ): load increases along every arc between customers
    for i in customers:
        for j in customers:
            if i != j:
                prob += u[j] >= u[i] + int(q[j]) - Q * (1 - x[i, j]), f"C4_mtz_{i}_{j}"

    # C5: load bounds
    for i in customers:
        prob += u[i] >= int(q[i]), f"C5_lo_{i}"
        prob += u[i] <= Q, f"C5_hi_{i}"

    return prob, x, u
    


