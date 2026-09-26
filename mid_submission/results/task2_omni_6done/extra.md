# Task 2 add-on analyses

500 problems; i.i.d. reference = 64 chains/problem.

## A. Tree vs. i.i.d., paired per problem (95% bootstrap CI)

PassRate difference in points. *Same leaves*: i.i.d. pass@k with k = #leaves (only if ≤ 64). *Same tokens*: i.i.d. pass@k at the k whose cost equals the tree's tokens on that problem.

| method | leaves | tree PassRate | Δ vs i.i.d. same leaves | Δ vs i.i.d. same tokens | matched k (mean) |
|---|---|---|---|---|---|
| Random (4,3,1,1) | 16 | 44.8% | +0.5 [-1.6, +2.6] | +3.0 [+1.0, +5.1] | 11.0 |
| EPTree (4,3,1,1) | 16 | 44.6% | +0.3 [-2.0, +2.7] | +2.7 [+0.5, +5.0] | 11.3 |
| Random (6,2,1,2) | 30 | 46.0% | -2.4 [-4.5, -0.5] | +0.4 [-1.6, +2.2] | 20.1 |
| EPTree (6,2,1,2) | 30 | 46.6% | -1.8 [-3.7, -0.0] | +0.9 [-0.9, +2.7] | 20.4 |
| EPTree (8,4,2,2) | 136 | 57.6% | n/a (> 64) | +4.3 [+1.9, +6.7] | 76.1 |

## B. Sibling disagreement and C. continuation length, per fork

Reference: two independent chains from the prompt disagree in outcome **10.0%** of the time.

| method | forks | outcome differs from original | wrong→right | right→wrong | new branch tokens | original remainder | median new/remainder | new branch truncated |
|---|---|---|---|---|---|---|---|---|
| Random (4,3,1,1) | 6000 | 7.9% | 5.4% | 14.8% | 433 | 409 | 1.01 | 0.4% |
| EPTree (4,3,1,1) | 6000 | 8.0% | 5.3% | 15.6% | 448 | 437 | 1.01 | 0.3% |
| Random (6,2,1,2) | 12000 | 7.0% | 4.8% | 13.3% | 436 | 410 | 1.01 | 0.3% |
| EPTree (6,2,1,2) | 12000 | 7.7% | 5.0% | 15.3% | 440 | 432 | 1.01 | 0.3% |
| EPTree (8,4,2,2) | 64000 | 7.1% | 4.5% | 14.6% | 391 | 385 | 1.00 | 0.3% |

*wrong→right* = share of forks off a wrong original that end correct; *right→wrong* = the reverse.

**Disagreement by fork position** (pooled over the configs run with both methods: (4,3,1,1), (6,2,1,2)):

| relative fork position | EPTree | Random | EPTree share of its forks | Random share of its forks |
|---|---|---|---|---|
| 0.0–0.2 | 9.9% (n=4868) | 9.1% (n=4049) | 27% | 22% |
| 0.2–0.4 | 8.3% (n=4226) | 9.1% (n=4073) | 23% | 23% |
| 0.4–0.6 | 7.1% (n=3604) | 7.1% (n=4023) | 20% | 22% |
| 0.6–0.8 | 6.1% (n=3349) | 5.1% (n=3869) | 19% | 21% |
| 0.8–1.0 | 5.5% (n=1953) | 4.4% (n=1986) | 11% | 11% |

Random forks reweighted to EPTree's position mix: **7.5%** vs. EPTree **7.8%** (raw random 7.3%).

**Entropy vs. random forks, paired per problem (disagreement rate, points):**

| config | EPTree − Random [95% CI] |
|---|---|
| (4,3,1,1) | +0.1 [-0.8, +1.1] |
| (6,2,1,2) | +0.7 [-0.0, +1.4] |

## D. Estimating V(root) vs. the policy's pass@1

True pass@1 from 64 i.i.d. chains: **0.260**. Bias = mean(estimate − pass@1) over problems, paired bootstrap CI; MAE = mean |estimate − pass@1|.

| method | leaf-mean bias | child-mean bias | roots-only bias | leaf-mean MAE | child-mean MAE | roots-only MAE |
|---|---|---|---|---|---|---|
| Random (4,3,1,1) | +0.003 [-0.003, +0.010] | +0.003 [-0.004, +0.010] | +0.002 [-0.008, +0.013] | 0.037 | 0.039 | 0.056 |
| EPTree (4,3,1,1) | +0.001 [-0.006, +0.008] | +0.002 [-0.005, +0.009] | +0.002 [-0.008, +0.013] | 0.037 | 0.039 | 0.058 |
| Random (6,2,1,2) | -0.003 [-0.008, +0.003] | -0.002 [-0.007, +0.003] | -0.005 [-0.012, +0.003] | 0.031 | 0.030 | 0.045 |
| EPTree (6,2,1,2) | -0.004 [-0.008, +0.001] | -0.003 [-0.009, +0.003] | -0.002 [-0.010, +0.006] | 0.030 | 0.033 | 0.045 |
| EPTree (8,4,2,2) | -0.005 [-0.010, -0.001] | -0.002 [-0.007, +0.002] | +0.003 [-0.003, +0.010] | 0.023 | 0.025 | 0.039 |
