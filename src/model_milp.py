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

    return prob, x, u
    


