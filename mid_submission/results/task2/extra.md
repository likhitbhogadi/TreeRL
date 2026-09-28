# Task 2 add-on analyses

500 problems; i.i.d. reference = 64 chains/problem.

## A. Tree vs. i.i.d., paired per problem (95% bootstrap CI)

PassRate difference in points. *Same leaves*: i.i.d. pass@k with k = #leaves (only if ≤ 64). *Same tokens*: i.i.d. pass@k at the k whose cost equals the tree's tokens on that problem.

| method | leaves | tree PassRate | Δ vs i.i.d. same leaves | Δ vs i.i.d. same tokens | matched k (mean) |
|---|---|---|---|---|---|
| EPTree (2,1,1,1) | 4 | 34.6% | -0.9 [-2.9, +1.1] | +0.8 [-1.1, +2.8] | 3.2 |
| EPTree (2,3,1,1) | 8 | 40.2% | +0.2 [-2.1, +2.3] | +3.2 [+1.0, +5.4] | 5.4 |
| Random (4,1,1,1) | 8 | 39.4% | -0.6 [-2.7, +1.6] | +1.2 [-0.8, +3.4] | 6.2 |
| EPTree (4,1,1,1) | 8 | 39.8% | -0.2 [-2.3, +1.9] | +1.6 [-0.4, +3.7] | 6.3 |
| Random (4,3,1,1) | 16 | 42.2% | -2.2 [-4.3, -0.0] | +0.5 [-1.4, +2.5] | 10.6 |
| EPTree (4,3,1,1) | 16 | 43.0% | -1.4 [-3.4, +0.8] | +1.2 [-0.8, +3.4] | 10.9 |
| Random (6,2,1,2) | 30 | 48.8% | +0.5 [-1.7, +2.8] | +3.3 [+1.2, +5.7] | 19.1 |
| EPTree (6,2,1,2) | 30 | 48.8% | +0.5 [-1.7, +2.6] | +3.4 [+1.3, +5.5] | 19.8 |
| Random (8,4,2,2) | 136 | 57.2% | n/a (> 64) | +4.1 [+1.4, +6.9] | 69.8 |
| EPTree (8,4,2,2) | 136 | 56.4% | n/a (> 64) | +3.6 [+1.2, +5.9] | 72.4 |

## B. Sibling disagreement and C. continuation length, per fork

Reference: two independent chains from the prompt disagree in outcome **10.0%** of the time.

| method | forks | outcome differs from original | wrong→right | right→wrong | new branch tokens | original remainder | median new/remainder | new branch truncated |
|---|---|---|---|---|---|---|---|---|
| EPTree (2,1,1,1) | 1000 | 7.4% | 4.6% | 15.2% | 426 | 410 | 1.01 | 0.3% |
| EPTree (2,3,1,1) | 3000 | 7.8% | 4.7% | 16.4% | 410 | 394 | 1.01 | 0.1% |
| Random (4,1,1,1) | 2000 | 7.0% | 5.2% | 11.9% | 414 | 383 | 1.00 | 0.4% |
| EPTree (4,1,1,1) | 2000 | 7.8% | 5.7% | 13.7% | 419 | 400 | 1.02 | 0.2% |
| Random (4,3,1,1) | 6000 | 6.3% | 4.5% | 11.5% | 405 | 384 | 1.00 | 0.3% |
| EPTree (4,3,1,1) | 6000 | 7.2% | 4.9% | 13.6% | 421 | 409 | 1.01 | 0.1% |
| Random (6,2,1,2) | 12000 | 6.8% | 4.5% | 13.0% | 406 | 382 | 1.00 | 0.2% |
| EPTree (6,2,1,2) | 12000 | 7.6% | 4.9% | 15.5% | 419 | 400 | 1.01 | 0.2% |
| Random (8,4,2,2) | 64000 | 5.9% | 4.1% | 11.0% | 359 | 344 | 1.00 | 0.3% |
| EPTree (8,4,2,2) | 64000 | 6.8% | 4.5% | 13.5% | 363 | 355 | 1.01 | 0.2% |

*wrong→right* = share of forks off a wrong original that end correct; *right→wrong* = the reverse.

**Disagreement by fork position** (pooled over the configs run with both methods: (4,1,1,1), (4,3,1,1), (6,2,1,2), (8,4,2,2)):

| relative fork position | EPTree | Random | EPTree share of its forks | Random share of its forks |
|---|---|---|---|---|
| 0.0–0.2 | 10.0% (n=16842) | 9.4% (n=13378) | 20% | 16% |
| 0.2–0.4 | 7.4% (n=16227) | 8.0% (n=15180) | 19% | 18% |
| 0.4–0.6 | 7.3% (n=16182) | 6.7% (n=17153) | 19% | 20% |
| 0.6–0.8 | 6.2% (n=16792) | 5.2% (n=19126) | 20% | 23% |
| 0.8–1.0 | 4.3% (n=17957) | 2.4% (n=19163) | 21% | 23% |

Random forks reweighted to EPTree's position mix: **6.3%** vs. EPTree **7.0%** (raw random 6.1%).

**Entropy vs. random forks, paired per problem (disagreement rate, points):**

| config | EPTree − Random [95% CI] |
|---|---|
| (4,1,1,1) | +0.8 [-0.6, +2.2] |
| (4,3,1,1) | +0.9 [+0.1, +1.8] |
| (6,2,1,2) | +0.9 [+0.2, +1.6] |
| (8,4,2,2) | +0.9 [+0.5, +1.4] |

## D. Estimating V(root) vs. the policy's pass@1

True pass@1 from 64 i.i.d. chains: **0.259**. Bias = mean(estimate − pass@1) over problems, paired bootstrap CI; MAE = mean |estimate − pass@1|.

| method | leaf-mean bias | child-mean bias | roots-only bias | leaf-mean MAE | child-mean MAE | roots-only MAE |
|---|---|---|---|---|---|---|
| EPTree (2,1,1,1) | +0.001 [-0.011, +0.013] | +0.001 [-0.011, +0.013] | +0.004 [-0.011, +0.019] | 0.064 | 0.065 | 0.085 |
| EPTree (2,3,1,1) | -0.001 [-0.010, +0.009] | -0.003 [-0.013, +0.007] | +0.006 [-0.009, +0.021] | 0.052 | 0.054 | 0.084 |
| Random (4,1,1,1) | +0.001 [-0.007, +0.010] | -0.001 [-0.009, +0.008] | -0.003 [-0.014, +0.007] | 0.049 | 0.051 | 0.060 |
| EPTree (4,1,1,1) | +0.003 [-0.005, +0.011] | +0.002 [-0.006, +0.010] | -0.001 [-0.010, +0.009] | 0.047 | 0.048 | 0.058 |
| Random (4,3,1,1) | +0.001 [-0.006, +0.007] | -0.002 [-0.009, +0.005] | -0.002 [-0.012, +0.008] | 0.038 | 0.040 | 0.056 |
| EPTree (4,3,1,1) | -0.002 [-0.008, +0.006] | -0.000 [-0.007, +0.007] | -0.003 [-0.013, +0.007] | 0.038 | 0.040 | 0.057 |
| Random (6,2,1,2) | +0.002 [-0.004, +0.008] | +0.001 [-0.005, +0.007] | +0.003 [-0.006, +0.011] | 0.034 | 0.033 | 0.050 |
| EPTree (6,2,1,2) | -0.001 [-0.008, +0.005] | +0.001 [-0.006, +0.007] | +0.002 [-0.007, +0.011] | 0.035 | 0.034 | 0.050 |
| Random (8,4,2,2) | +0.003 [-0.002, +0.008] | +0.004 [-0.001, +0.010] | +0.002 [-0.005, +0.010] | 0.026 | 0.030 | 0.042 |
| EPTree (8,4,2,2) | -0.002 [-0.007, +0.002] | +0.002 [-0.003, +0.007] | +0.004 [-0.003, +0.012] | 0.026 | 0.028 | 0.041 |
