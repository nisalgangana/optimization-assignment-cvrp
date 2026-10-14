# Contributions

IT5082 Optimization Methods, CVRP assignment (group of 2).

## Nisal Gangana (MS26910222)

I worked on the shared base of the project and on the exact method.

**Shared code**

- `src/common.py`: the `Instance` and `Solution` classes, and the cost and
  feasibility functions that both methods use. Because both methods use the
  same functions, the MILP and SA results are measured in the same way.
- `src/loader.py`: reads the CVRPLIB `.vrp` files, builds the rounded distance
  matrix, and creates the synthetic instances for the scalability test.
- `src/download_instances.py` and the five benchmark instances in
  `data/instances/`.

**Exact method (MILP)**

- `src/model_milp.py`: the MTZ model (constraints C1 to C5) written in PuLP and
  solved with CBC.
- The `solve()` function: time limit, turning the solver status into
  optimal / timeout, reading the lower bound from the CBC log, and building the
  routes from the x variables.
- Setup fixes for PuLP 4, which no longer includes CBC, and for the macOS
  code-signing problem that stopped CBC from starting.
- First MILP test runs on P-n16-k8, E-n22-k4 and A-n32-k5.

**Notebook and documentation**

- `notebooks/main.ipynb`: data, model size, MILP results and the final
  comparison with SA.
- `README.md`: setup steps, how to run each script, and the results tables.
- Report sections 1 to 4 (introduction, data, mathematical model and exact
  method results).

Pull requests: #1, #2, #3, #4, #7, #8.

## Ayesh Lakshan (MS26901626)

