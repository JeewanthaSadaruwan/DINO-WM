# point_maze corrected evaluation

Previous evaluation = 50 repeated runs of ONE task (eval_seed was always 1); it is not a valid 50-task benchmark.

| Metric | Previous evaluation (1 task x 50) | Corrected evaluation |
|---|---|---|
| Total runs | 50 | 50 |
| Distinct start/goal tasks | 1 | 50 |
| Successful runs | 50 | 49 |
| Success rate | 100% | 98% |
| Mean final state distance | 1.816 | 4.700 |
| Mean final visual distance | 0.468 | 0.502 |
| Mean final proprio distance | 0.693 | 1.621 |

Paper (DINO-WM, MPC, 50 tasks): 98%. Corrected: 98% (49/50).

Failed tasks (eval seeds): [45]
