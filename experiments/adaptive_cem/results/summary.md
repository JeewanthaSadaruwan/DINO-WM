# Adaptive CEM vs Original CEM — PointMaze (50 paired tasks)

| Metric | Original CEM | Adaptive CEM | Change |
|---|---|---|---|
| Evaluated tasks | 50 | 50 | paired |
| Successful tasks | 49 | 50 | +1 |
| Success rate | 98.0% | 100.0% | +2.0 pp |
| Total CEM iterations | 452 | 220 | -232 (-51.3%) |
| Average CEM iterations per planning call | 5.72 | 3.10 | -2.62 (-45.8%) |
| Average MPC replanning calls per task | 1.58 | 1.42 | -0.16 (-10.1%) |
| Total candidate sequences evaluated | 135,600 | 66,000 | -69,600 (-51.3%) |
| Total predictor forward calls (batched) | 2,260 | 1,100 | -1,160 (-51.3%) |
| Average planning time per task | 42.68 s | 20.95 s | -21.72 s (-50.9%) |
| Total planning time | 2133.9 s | 1047.7 s | -1,086.2 s (-50.9%) |
| Mean final state distance (pos+vel) | 4.700 | 4.551 | -0.149 (-3.2%) |
| Convergence-triggered stops (calls) | 0 | 20 |  |
| Tasks hitting MPC step cap | 1 | 0 |  |

## Computation

- Total CEM iterations was reduced by 51.3% (95% bootstrap CI 11.1% to 72.2%).
- Total planning time was reduced by 50.9% (95% bootstrap CI 10.9% to 71.9%).
- Total candidate sequences evaluated was reduced by 51.3% (95% bootstrap CI 11.1% to 72.2%).
- Adaptive stopping ended 20 of 71 planning calls; MPC replanning calls per task: 1.58 → 1.42 (all replanning cost is included in the totals).
- Per task: adaptive used fewer CEM iterations on 9, more on 2, and the same number on 39 of 50 tasks (median planning-time difference -0.00 s). Savings are concentrated in a few tasks.
- Sensitivity: excluding tasks where either planner hit the MPC step cap ([45]), R_I = 20.6% and R_T = 20.3% over 49 tasks. Capped tasks inflate totals, because the cap (not the planner) sets their cost.

## Navigation performance

- Success: 49/50 → 50/50 (+2.0 percentage points).
- Paired outcomes: solved only by Original = 0, only by Adaptive = 1; exact McNemar p = 1.000.
- Mean final state distance: 4.700 → 4.551.
- No statistically significant change in success was detected. With this sample size, small differences cannot be ruled out, so this is not proof that the two planners are equivalent.

## Notes

- Planning time = GPU-synchronised world-model inference and CEM updates. It excludes the original per-iteration simulator success check, rendering, video writing and model loading.
- Final state distance is the env's full-state distance (position and velocity); success uses position only (< 0.5).
- Task inputs were verified identical for both planners on all 50 paired tasks.
- Excluded runs: none.
