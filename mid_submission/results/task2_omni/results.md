# Task 2 results: EPTree replication

500 problems common to all runs; i.i.d. baseline from `b0_64.jsonl` (64 chains/problem).

## i.i.d. chains (B0), unbiased pass@k

| k | tokens/problem | PassRate | distinct answers |
|---|---|---|---|
| 1 | 741 | 26.0% | 0.99 |
| 2 | 1482 | 31.0% | 1.58 |
| 4 | 2966 | 35.5% | 2.42 |
| 8 | 5905 | 39.9% | 3.87 |
| 16 | 11833 | 44.3% | 6.04 |
| 32 | 23695 | 48.9% | 9.32 |
| 64 | 47408 | 53.4% | 14.42 |

## Tree configs (Fig. 4 / Fig. 5 data)

| method | (M,N,L,T) | leaves | tokens/problem | PassRate | i.i.d. PassRate at same tokens | mean acc | distinct answers | mixed-outcome trees |
|---|---|---|---|---|---|---|---|---|
| EPTree (B2) | (2,1,1,1) | 4 | 2363 | 34.4% | 34.0% | 0.261 | 2.39 | 16% |
| EPTree (B2) | (2,3,1,1) | 8 | 4141 | 40.2% | 37.6% | 0.261 | 3.62 | 26% |
| Random (B1) | (4,1,1,1) | 8 | 4691 | 38.8% | 38.4% | 0.260 | 3.60 | 24% |
| EPTree (B2) | (4,1,1,1) | 8 | 4713 | 40.0% | 38.5% | 0.266 | 3.68 | 24% |
| Random (B1) | (4,3,1,1) | 16 | 8143 | 44.8% | 41.9% | 0.263 | 5.53 | 32% |
| EPTree (B2) | (4,3,1,1) | 16 | 8353 | 44.6% | 42.1% | 0.260 | 5.62 | 32% |
| Random (B1) | (6,2,1,2) | 30 | 14895 | 46.0% | 45.8% | 0.257 | 8.07 | 36% |
| EPTree (B2) | (6,2,1,2) | 30 | 14981 | 46.6% | 45.9% | 0.256 | 8.24 | 36% |
| EPTree (B2) | (8,4,2,2) | 136 | 55880 | 57.6% | nan% | 0.254 | 21.69 | 50% |
| Random (B1) | (8,4,2,2) | 136 | 55986 | 55.4% | nan% | 0.260 | 20.60 | 48% |

## Table 2: entropy-guided vs. random forking (same M,N,L,T)

| (M,N,L,T) | PassRate B2 | PassRate B1 | Δ PassRate [95% CI] | acc B2 | acc B1 | distinct B2 | distinct B1 | tokens B2 | tokens B1 |
|---|---|---|---|---|---|---|---|---|---|
| (4,1,1,1) | 40.0% | 38.8% | +1.2 [-1.2, +3.6] | 0.266 | 0.260 | 3.68 | 3.60 | 4713 | 4691 |
| (4,3,1,1) | 44.6% | 44.8% | -0.2 [-3.0, +2.6] | 0.260 | 0.263 | 5.62 | 5.53 | 8353 | 8143 |
| (6,2,1,2) | 46.6% | 46.0% | +0.6 [-1.8, +3.0] | 0.256 | 0.257 | 8.24 | 8.07 | 14981 | 14895 |
| (8,4,2,2) | 57.6% | 55.4% | +2.2 [-0.2, +4.8] | 0.254 | 0.260 | 21.69 | 20.60 | 55880 | 55986 |

## Fig. 7: top-10 forking tokens (EPTree, all configs pooled)

| rank | token | share of forks |
|---|---|---|
| 1 | `␣the` | 1.8% |
| 2 | `␣\(` | 1.6% |
| 3 | `␣\` | 1.0% |
| 4 | `,` | 0.9% |
| 5 | `␣` | 0.9% |
| 6 | `␣a` | 0.7% |
| 7 | `.` | 0.7% |
| 8 | `␣we` | 0.7% |
| 9 | `␣and` | 0.7% |
| 10 | `␣in` | 0.6% |

## Fig. 8: fork positions and surprisal

| method | forks | mean relative position | median relative position | mean surprisal at fork token |
|---|---|---|---|---|
| EPTree (B2) | 88000 | 0.47 | 0.48 | 4.79 |
| Random (B1) | 84000 | 0.49 | 0.51 | 0.20 |
