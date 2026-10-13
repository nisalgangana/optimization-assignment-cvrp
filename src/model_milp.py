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

import re
import tempfile

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
    
def _read_lower_bound(log_path): #read the lower bound from the CBC log file (if it exists)
    try:
        text = Path(log_path).read_text() #read the log file
    except OSError:
        return None 
    m = re.search(r"Lower bound:\s*([-+\d.eE]+)", text) #search for the line containing the lower bound
    return float(m.group(1)) if m else None


def _extract_routes(inst, x): #extract the routes from the solution variables x[i,j] (which are 0/1)
    succ = {i: j for (i, j), var in x.items() if i != 0 and var.value() > 0.5}
    routes = [] #list to store the extracted routes
    for j in inst.customers: 
        if x[0, j].value() > 0.5:
            route = [j]
            while succ[route[-1]] != 0: #follow the successor chain until we reach the depot (0)
                route.append(succ[route[-1]])
            routes.append(route)
    return routes


def solve(inst: Instance, time_limit: float = 300.0, seed: int = 0, **kw) -> Solution:
    prob, x, u = build_model(inst)

    # solve with CBC; the log goes to a temporary file so the bound can be read
    with tempfile.TemporaryDirectory() as tmp: #create a temporary directory to store the log file
        log_path = str(Path(tmp) / "cbc.log")
        solver = pulp.COIN_CMD(msg=False, timeLimit=time_limit,
                               randomSeed=seed, logPath=log_path)
        t0 = time.perf_counter() #start the timer
        stats = prob.solve(solver) 
        runtime = time.perf_counter() - t0 #stop the timer and calculate the runtime
        bound = stats.best_bound
        if bound is None:
            bound = _read_lower_bound(log_path)

    # map the solver result to our status names
    S = pulp.LpSolveStatus
    if stats.status == S.Optimal:
        status = "optimal"
    elif stats.status == S.Infeasible:
        status = "infeasible"
    elif stats.status == S.TimeLimit:
        status = "timeout"
    else:
        status = "feasible" if stats.has_solution else "infeasible"

    # no solution found (e.g. time ran out before the first route set)
    if not stats.has_solution:
        return Solution(routes=[], cost=float("inf"), runtime=runtime,
                        method="milp", status=status, bound=bound,
                        extra={"seed": seed, "time_limit": time_limit})

    routes = _extract_routes(inst, x) 
    cost = common.solution_cost(inst, routes)
    ok, why = common.is_feasible(inst, routes)
    assert ok, f"MILP returned an infeasible solution: {why}"
    if status == "optimal":
        bound = cost  # optimum proved, so the bound equals the cost

    return Solution(routes=routes, cost=cost, runtime=runtime, method="milp",
                    status=status, bound=bound,
                    extra={"seed": seed, "time_limit": time_limit,
                           "solver_status": stats.status.name})


if __name__ == "__main__": #solve a CVRP instance from the command line
    path = sys.argv[1] if len(sys.argv) > 1 else "data/instances/P-n16-k8.vrp"
    limit = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0 #time limit in seconds
    inst = load_cvrplib(path) #load the CVRP instance from the given path
    sol = solve(inst, time_limit=limit) 
    print(common.describe(inst, sol))
    print(f"  lower bound = {sol.bound}") 

