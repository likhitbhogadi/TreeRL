# Task 2 add-on analyses

500 problems; i.i.d. reference = 64 chains/problem.

## A. Tree vs. i.i.d., paired per problem (95% bootstrap CI)

PassRate difference in points. *Same leaves*: i.i.d. pass@k with k = #leaves (only if ≤ 64). *Same tokens*: i.i.d. pass@k at the k whose cost equals the tree's tokens on that problem.

| method | leaves | tree PassRate | Δ vs i.i.d. same leaves | Δ vs i.i.d. same tokens | matched k (mean) |
|---|---|---|---|---|---|
| EPTree (2,1,1,1) | 4 | 34.4% | -1.1 [-2.9, +0.8] | +0.6 [-1.2, +2.5] | 3.2 |
| EPTree (2,3,1,1) | 8 | 40.2% | +0.3 [-1.7, +2.4] | +2.7 [+0.8, +4.8] | 5.6 |
| Random (4,1,1,1) | 8 | 38.8% | -1.1 [-2.7, +0.6] | +0.5 [-1.1, +2.2] | 6.3 |
| EPTree (4,1,1,1) | 8 | 40.0% | +0.1 [-1.7, +2.0] | +1.8 [-0.1, +3.7] | 6.4 |
| Random (4,3,1,1) | 16 | 44.8% | +0.5 [-1.6, +2.6] | +3.0 [+1.0, +5.1] | 11.0 |
| EPTree (4,3,1,1) | 16 | 44.6% | +0.3 [-2.0, +2.7] | +2.7 [+0.5, +5.0] | 11.3 |
| Random (6,2,1,2) | 30 | 46.0% | -2.4 [-4.5, -0.5] | +0.4 [-1.6, +2.2] | 20.1 |
| EPTree (6,2,1,2) | 30 | 46.6% | -1.8 [-3.7, -0.0] | +0.9 [-0.9, +2.7] | 20.4 |
| EPTree (8,4,2,2) | 136 | 57.6% | n/a (> 64) | +4.3 [+1.9, +6.7] | 76.1 |
| Random (8,4,2,2) | 136 | 55.4% | n/a (> 64) | +2.0 [-0.8, +4.8] | 75.3 |

## B. Sibling disagreement and C. continuation length, per fork

Reference: two independent chains from the prompt disagree in outcome **10.0%** of the time.

| method | forks | outcome differs from original | wrong→right | right→wrong | new branch tokens | original remainder | median new/remainder | new branch truncated |
|---|---|---|---|---|---|---|---|---|
| EPTree (2,1,1,1) | 1000 | 7.6% | 5.5% | 13.6% | 446 | 426 | 1.02 | 0.2% |
| EPTree (2,3,1,1) | 3000 | 7.7% | 4.8% | 15.7% | 444 | 436 | 1.01 | 0.2% |
| Random (4,1,1,1) | 2000 | 6.1% | 3.6% | 13.1% | 434 | 413 | 1.01 | 0.5% |
| EPTree (4,1,1,1) | 2000 | 6.5% | 4.2% | 12.7% | 443 | 432 | 1.01 | 0.4% |
| Random (4,3,1,1) | 6000 | 7.9% | 5.4% | 14.8% | 433 | 409 | 1.01 | 0.4% |
| EPTree (4,3,1,1) | 6000 | 8.0% | 5.3% | 15.6% | 448 | 437 | 1.01 | 0.3% |
| Random (6,2,1,2) | 12000 | 7.0% | 4.8% | 13.3% | 436 | 410 | 1.01 | 0.3% |
| EPTree (6,2,1,2) | 12000 | 7.7% | 5.0% | 15.3% | 440 | 432 | 1.01 | 0.3% |
| EPTree (8,4,2,2) | 64000 | 7.1% | 4.5% | 14.6% | 391 | 385 | 1.00 | 0.3% |
| Random (8,4,2,2) | 64000 | 6.5% | 4.7% | 12.0% | 391 | 378 | 1.00 | 0.5% |

*wrong→right* = share of forks off a wrong original that end correct; *right→wrong* = the reverse.

**Disagreement by fork position** (pooled over the configs run with both methods: (4,1,1,1), (4,3,1,1), (6,2,1,2), (8,4,2,2)):

| relative fork position | EPTree | Random | EPTree share of its forks | Random share of its forks |
|---|---|---|---|---|
| 0.0–0.2 | 9.5% (n=17066) | 9.1% (n=14299) | 20% | 17% |
| 0.2–0.4 | 7.8% (n=17359) | 8.5% (n=16924) | 21% | 20% |
| 0.4–0.6 | 7.2% (n=17815) | 7.0% (n=19277) | 21% | 23% |
| 0.6–0.8 | 6.2% (n=19624) | 5.1% (n=21652) | 23% | 26% |
| 0.8–1.0 | 5.1% (n=12136) | 3.7% (n=11848) | 14% | 14% |

Random forks reweighted to EPTree's position mix: **6.8%** vs. EPTree **7.3%** (raw random 6.7%).

**Entropy vs. random forks, paired per problem (disagreement rate, points):**

| config | EPTree − Random [95% CI] |
|---|---|
| (4,1,1,1) | +0.4 [-0.9, +1.6] |
| (4,3,1,1) | +0.1 [-0.8, +1.1] |
| (6,2,1,2) | +0.7 [-0.0, +1.4] |
| (8,4,2,2) | +0.6 [+0.2, +1.0] |

## D. Estimating V(root) vs. the policy's pass@1

True pass@1 from 64 i.i.d. chains: **0.260**. Bias = mean(estimate − pass@1) over problems, paired bootstrap CI; MAE = mean |estimate − pass@1|.

| method | leaf-mean bias | child-mean bias | roots-only bias | leaf-mean MAE | child-mean MAE | roots-only MAE |
|---|---|---|---|---|---|---|
| EPTree (2,1,1,1) | +0.001 [-0.010, +0.012] | +0.002 [-0.009, +0.014] | -0.002 [-0.015, +0.013] | 0.062 | 0.063 | 0.076 |
| EPTree (2,3,1,1) | +0.002 [-0.008, +0.011] | +0.006 [-0.003, +0.015] | +0.006 [-0.008, +0.020] | 0.051 | 0.052 | 0.078 |
| Random (4,1,1,1) | +0.000 [-0.007, +0.008] | +0.002 [-0.006, +0.010] | +0.004 [-0.005, +0.014] | 0.041 | 0.044 | 0.051 |
| EPTree (4,1,1,1) | +0.007 [-0.001, +0.015] | +0.006 [-0.003, +0.014] | +0.008 [-0.001, +0.019] | 0.045 | 0.045 | 0.053 |
| Random (4,3,1,1) | +0.003 [-0.003, +0.010] | +0.003 [-0.004, +0.010] | +0.002 [-0.008, +0.013] | 0.037 | 0.039 | 0.056 |
| EPTree (4,3,1,1) | +0.001 [-0.006, +0.008] | +0.002 [-0.005, +0.009] | +0.002 [-0.008, +0.013] | 0.037 | 0.039 | 0.058 |
| Random (6,2,1,2) | -0.003 [-0.008, +0.003] | -0.002 [-0.007, +0.003] | -0.005 [-0.012, +0.003] | 0.031 | 0.030 | 0.045 |
| EPTree (6,2,1,2) | -0.004 [-0.008, +0.001] | -0.003 [-0.009, +0.003] | -0.002 [-0.010, +0.006] | 0.030 | 0.033 | 0.045 |
| EPTree (8,4,2,2) | -0.005 [-0.010, -0.001] | -0.002 [-0.007, +0.002] | +0.003 [-0.003, +0.010] | 0.023 | 0.025 | 0.039 |
| Random (8,4,2,2) | +0.001 [-0.003, +0.005] | -0.000 [-0.004, +0.004] | -0.003 [-0.010, +0.004] | 0.022 | 0.024 | 0.040 |
