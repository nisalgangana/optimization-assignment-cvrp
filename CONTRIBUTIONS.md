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

I worked on the heuristic method, the experiments and the figures.

**Heuristic method**

- `src/model_heuristic.py`: a capacity-aware nearest-neighbour greedy
  construction, and simulated annealing that improves it with three moves
  (relocate, swap and 2-opt). Every move checks capacity with
  `common.route_load`, and every cost comes from `common.solution_cost`.
- SA details: start temperature from the average change of 100 random moves,
  geometric cooling (T = 0.995 T), keeping the best plan separately from the
  current one, and a penalty that empties an extra route when greedy uses more
  trucks than allowed (needed for A-n45-k6).
- `solve(inst, time_limit, seed)` follows the same interface as the MILP, so
  both methods can be run the same way.

**Experiments and results**

- `src/evaluate.py`: runs the MILP (600 s) and SA (5 seeds) on the five
  benchmark instances and on synthetic instances with 10, 20, 40 and 80
  customers (scalability study), and writes `results/results.csv`.
- `src/tuning.py`: a small sensitivity study of the cooling rate and the start
  temperature, saved in `results/tuning.csv`.
- `src/plots.py`: the result figures (cost comparison, runtime and gap against
  size, SA convergence) and `plot_routes()`, which draws the routes of either
  method and is also used in `main.ipynb`.

**Notebook, report and video**

- `notebooks/heuristic_method.ipynb`: SA on three instances with the
  convergence plot and a route map.
- Report sections 5 to 7 (heuristic method, results, discussion and future
  work), and video minutes 7:30 to 15:00.

Pull requests: #5, #6.
