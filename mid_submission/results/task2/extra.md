# Task 2 add-on analyses

243 problems; i.i.d. reference = 64 chains/problem.

## A. Tree vs. i.i.d., paired per problem (95% bootstrap CI)

PassRate difference in points. *Same leaves*: i.i.d. pass@k with k = #leaves (only if ≤ 64). *Same tokens*: i.i.d. pass@k at the k whose cost equals the tree's tokens on that problem.

| method | leaves | tree PassRate | Δ vs i.i.d. same leaves | Δ vs i.i.d. same tokens | matched k (mean) |
|---|---|---|---|---|---|
| EPTree (4,1,3,1) | 16 | 90.1% | -4.9 [-8.6, -1.6] | -1.2 [-4.7, +1.9] | 10.8 |
| Random (4,3,1,1) | 16 | 92.2% | -2.8 [-6.2, +0.2] | +0.3 [-3.0, +3.4] | 11.0 |
| EPTree (4,3,1,1) | 16 | 93.8% | -1.2 [-4.0, +1.3] | +1.8 [-1.0, +4.3] | 11.5 |
| EPTree (8,1,1,1) | 16 | 94.2% | -0.8 [-3.4, +1.6] | +0.8 [-1.7, +3.4] | 13.1 |
| Random (6,2,1,2) | 30 | 97.9% | -0.1 [-2.1, +2.0] | +1.7 [-0.2, +3.8] | 20.0 |
| EPTree (6,2,1,2) | 30 | 97.9% | -0.1 [-1.7, +1.4] | +1.5 [-0.1, +3.0] | 21.0 |
| EPTree (8,1,7,1) | 64 | 99.6% | +0.0 [-1.2, +1.2] | +1.2 [+0.0, +2.6] | 36.0 |
| EPTree (8,7,1,1) | 64 | 98.4% | -1.2 [-2.9, +0.0] | -0.6 [-2.2, +0.7] | 43.6 |
| Random (16,3,1,1) | 64 | 99.2% | -0.4 [-2.1, +0.8] | +0.1 [-1.4, +1.5] | 44.0 |
| EPTree (16,3,1,1) | 64 | 100.0% | +0.4 [+0.0, +1.2] | +1.0 [+0.3, +2.1] | 46.5 |

## B. Sibling disagreement and C. continuation length, per fork

Reference: two independent chains from the prompt disagree in outcome **34.4%** of the time.

| method | forks | outcome differs from original | wrong→right | right→wrong | new branch tokens | original remainder | median new/remainder | new branch truncated |
|---|---|---|---|---|---|---|---|---|
| EPTree (4,1,3,1) | 2916 | 21.8% | 20.3% | 23.8% | 455 | 460 | 1.00 | 0.2% |
| Random (4,3,1,1) | 2916 | 25.0% | 23.1% | 27.1% | 475 | 457 | 1.01 | 0.3% |
| EPTree (4,3,1,1) | 2916 | 25.6% | 21.8% | 30.2% | 512 | 509 | 1.01 | 0.1% |
| EPTree (8,1,1,1) | 1944 | 25.1% | 23.5% | 27.1% | 522 | 515 | 1.01 | 0.4% |
| Random (6,2,1,2) | 5832 | 20.5% | 19.2% | 22.1% | 477 | 451 | 1.00 | 0.2% |
| EPTree (6,2,1,2) | 5832 | 24.8% | 22.3% | 27.7% | 519 | 505 | 1.01 | 0.2% |
| EPTree (8,1,7,1) | 13608 | 20.1% | 17.4% | 23.9% | 409 | 410 | 0.99 | 0.3% |
| EPTree (8,7,1,1) | 13608 | 25.4% | 22.7% | 28.7% | 527 | 506 | 1.01 | 0.2% |
| Random (16,3,1,1) | 11664 | 21.7% | 19.6% | 24.1% | 480 | 450 | 1.01 | 0.4% |
| EPTree (16,3,1,1) | 11664 | 23.7% | 20.3% | 27.9% | 522 | 509 | 1.01 | 0.3% |

*wrong→right* = share of forks off a wrong original that end correct; *right→wrong* = the reverse.

**Disagreement by fork position** (pooled over the configs run with both methods: (16,3,1,1), (4,3,1,1), (6,2,1,2)):

| relative fork position | EPTree | Random | EPTree share of its forks | Random share of its forks |
|---|---|---|---|---|
| 0.0–0.2 | 32.0% (n=6491) | 33.0% (n=4421) | 32% | 22% |
| 0.2–0.4 | 27.7% (n=4461) | 28.4% (n=4551) | 22% | 22% |
| 0.4–0.6 | 22.8% (n=3865) | 20.3% (n=4603) | 19% | 23% |
| 0.6–0.8 | 15.6% (n=3568) | 12.9% (n=4585) | 17% | 22% |
| 0.8–1.0 | 10.1% (n=2027) | 7.7% (n=2252) | 10% | 11% |

Random forks reweighted to EPTree's position mix: **23.6%** vs. EPTree **24.3%** (raw random 21.8%).

**Entropy vs. random forks, paired per problem (disagreement rate, points):**

| config | EPTree − Random [95% CI] |
|---|---|
| (16,3,1,1) | +2.0 [+0.9, +3.2] |
| (4,3,1,1) | +0.6 [-1.6, +2.8] |
| (6,2,1,2) | +4.3 [+2.4, +6.0] |

## D. Estimating V(root) vs. the policy's pass@1

True pass@1 from 64 i.i.d. chains: **0.458**. Bias = mean(estimate − pass@1) over problems, paired bootstrap CI; MAE = mean |estimate − pass@1|.

| method | leaf-mean bias | child-mean bias | roots-only bias | leaf-mean MAE | child-mean MAE | roots-only MAE |
|---|---|---|---|---|---|---|
| EPTree (4,1,3,1) | -0.015 [-0.035, +0.005] | -0.004 [-0.025, +0.017] | -0.019 [-0.045, +0.009] | 0.120 | 0.125 | 0.167 |
| Random (4,3,1,1) | +0.005 [-0.014, +0.024] | +0.014 [-0.008, +0.035] | +0.007 [-0.021, +0.035] | 0.121 | 0.133 | 0.170 |
| EPTree (4,3,1,1) | -0.020 [-0.037, -0.002] | -0.008 [-0.027, +0.011] | -0.007 [-0.032, +0.019] | 0.115 | 0.120 | 0.166 |
| EPTree (8,1,1,1) | -0.014 [-0.029, +0.001] | -0.014 [-0.029, +0.001] | -0.020 [-0.040, -0.001] | 0.094 | 0.093 | 0.121 |
| Random (6,2,1,2) | -0.007 [-0.022, +0.008] | -0.008 [-0.023, +0.007] | -0.014 [-0.035, +0.008] | 0.093 | 0.096 | 0.132 |
| EPTree (6,2,1,2) | -0.007 [-0.023, +0.008] | -0.003 [-0.020, +0.013] | -0.005 [-0.026, +0.017] | 0.097 | 0.101 | 0.143 |
| EPTree (8,1,7,1) | -0.037 [-0.050, -0.025] | -0.013 [-0.026, +0.001] | -0.017 [-0.036, +0.002] | 0.089 | 0.082 | 0.120 |
| EPTree (8,7,1,1) | -0.011 [-0.022, -0.000] | -0.010 [-0.022, +0.002] | -0.006 [-0.026, +0.012] | 0.067 | 0.080 | 0.117 |
| Random (16,3,1,1) | -0.001 [-0.012, +0.009] | -0.005 [-0.016, +0.006] | +0.003 [-0.012, +0.016] | 0.065 | 0.068 | 0.092 |
| EPTree (16,3,1,1) | -0.018 [-0.028, -0.008] | -0.015 [-0.026, -0.004] | -0.008 [-0.022, +0.006] | 0.066 | 0.071 | 0.088 |
