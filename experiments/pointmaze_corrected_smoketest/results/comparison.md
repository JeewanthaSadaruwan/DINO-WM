# point_maze corrected evaluation

Previous evaluation = 50 repeated runs of ONE task (eval_seed was always 1); it is not a valid 50-task benchmark.

| Metric | Previous evaluation (1 task x 50) | Corrected evaluation |
|---|---|---|
| Total runs | 50 | 3 |
| Distinct start/goal tasks | 1 | 3 |
| Successful runs | 50 | 3 |
| Success rate | 100% | 100% |
| Mean final state distance | 1.816 | 3.222 |
| Mean final visual distance | 0.468 | 0.418 |
| Mean final proprio distance | 0.693 | 1.039 |

Paper (DINO-WM, MPC, 50 tasks): 98%. Corrected: 100% (3/3).
