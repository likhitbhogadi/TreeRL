# Task 2 results: EPTree replication

500 problems common to all runs; i.i.d. baseline from `b0_64.jsonl` (64 chains/problem).

## i.i.d. chains (B0), unbiased pass@k

| k | tokens/problem | PassRate | distinct answers |
|---|---|---|---|
| 1 | 745 | 25.9% | 0.99 |
| 2 | 1478 | 30.9% | 1.59 |
| 4 | 2940 | 35.5% | 2.51 |
| 8 | 5941 | 40.0% | 3.92 |
| 16 | 11854 | 44.4% | 6.00 |
| 32 | 23657 | 48.7% | 9.30 |
| 64 | 47344 | 53.2% | 14.41 |

## Tree configs (Fig. 4 / Fig. 5 data)

| method | (M,N,L,T) | leaves | tokens/problem | PassRate | i.i.d. PassRate at same tokens | mean acc | distinct answers | mixed-outcome trees |
|---|---|---|---|---|---|---|---|---|
| EPTree (B2) | (2,1,1,1) | 4 | 2341 | 34.6% | 34.0% | 0.260 | 2.32 | 17% |
| EPTree (B2) | (2,3,1,1) | 8 | 3925 | 40.2% | 37.4% | 0.258 | 3.56 | 27% |
| Random (B1) | (4,1,1,1) | 8 | 4613 | 39.4% | 38.4% | 0.260 | 3.63 | 24% |
| EPTree (B2) | (4,1,1,1) | 8 | 4622 | 39.8% | 38.4% | 0.262 | 3.65 | 25% |
| Random (B1) | (4,3,1,1) | 16 | 7854 | 42.2% | 41.8% | 0.260 | 5.38 | 30% |
| EPTree (B2) | (4,3,1,1) | 16 | 8031 | 43.0% | 41.9% | 0.258 | 5.53 | 32% |
| Random (B1) | (6,2,1,2) | 30 | 14163 | 48.8% | 45.5% | 0.261 | 8.07 | 38% |
| EPTree (B2) | (6,2,1,2) | 30 | 14457 | 48.8% | 45.6% | 0.258 | 8.41 | 39% |
| Random (B1) | (8,4,2,2) | 136 | 51894 | 57.2% | nan% | 0.263 | 19.92 | 48% |
| EPTree (B2) | (8,4,2,2) | 136 | 52355 | 56.4% | nan% | 0.257 | 21.07 | 48% |

## Table 2: entropy-guided vs. random forking (same M,N,L,T)

| (M,N,L,T) | PassRate B2 | PassRate B1 | Δ PassRate [95% CI] | acc B2 | acc B1 | distinct B2 | distinct B1 | tokens B2 | tokens B1 |
|---|---|---|---|---|---|---|---|---|---|
| (4,1,1,1) | 39.8% | 39.4% | +0.4 [-2.0, +2.8] | 0.262 | 0.260 | 3.65 | 3.63 | 4622 | 4613 |
| (4,3,1,1) | 43.0% | 42.2% | +0.8 [-1.6, +3.4] | 0.258 | 0.260 | 5.53 | 5.38 | 8031 | 7854 |
| (6,2,1,2) | 48.8% | 48.8% | +0.0 [-2.4, +2.4] | 0.258 | 0.261 | 8.41 | 8.07 | 14457 | 14163 |
| (8,4,2,2) | 56.4% | 57.2% | -0.8 [-3.2, +1.6] | 0.257 | 0.263 | 21.07 | 19.92 | 52355 | 51894 |

## Fig. 7: top-10 forking tokens (EPTree, all configs pooled)

| rank | token | share of forks |
|---|---|---|
| 1 | `␣the` | 1.9% |
| 2 | `␣\(` | 1.4% |
| 3 | `␣\` | 1.0% |
| 4 | `,` | 0.9% |
| 5 | `␣` | 0.8% |
| 6 | `␣and` | 0.7% |
| 7 | `␣a` | 0.6% |
| 8 | `␣we` | 0.6% |
| 9 | `.` | 0.6% |
| 10 | `1` | 0.6% |

## Fig. 8: fork positions and surprisal

| method | forks | mean relative position | median relative position | mean surprisal at fork token |
|---|---|---|---|---|
| EPTree (B2) | 88000 | 0.50 | 0.51 | 4.95 |
| Random (B1) | 84000 | 0.54 | 0.56 | 0.19 |
