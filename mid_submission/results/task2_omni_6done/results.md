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
| Random (B1) | (4,3,1,1) | 16 | 8143 | 44.8% | 41.9% | 0.263 | 5.53 | 32% |
| EPTree (B2) | (4,3,1,1) | 16 | 8353 | 44.6% | 42.1% | 0.260 | 5.62 | 32% |
| Random (B1) | (6,2,1,2) | 30 | 14895 | 46.0% | 45.8% | 0.257 | 8.07 | 36% |
| EPTree (B2) | (6,2,1,2) | 30 | 14981 | 46.6% | 45.9% | 0.256 | 8.24 | 36% |
| EPTree (B2) | (8,4,2,2) | 136 | 55880 | 57.6% | nan% | 0.254 | 21.69 | 50% |

## Table 2: entropy-guided vs. random forking (same M,N,L,T)

| (M,N,L,T) | PassRate B2 | PassRate B1 | Δ PassRate [95% CI] | acc B2 | acc B1 | distinct B2 | distinct B1 | tokens B2 | tokens B1 |
|---|---|---|---|---|---|---|---|---|---|
| (4,3,1,1) | 44.6% | 44.8% | -0.2 [-3.0, +2.6] | 0.260 | 0.263 | 5.62 | 5.53 | 8353 | 8143 |
| (6,2,1,2) | 46.6% | 46.0% | +0.6 [-1.8, +3.0] | 0.256 | 0.257 | 8.24 | 8.07 | 14981 | 14895 |

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
| 9 | `␣and` | 0.6% |
| 10 | `␣in` | 0.6% |

## Fig. 8: fork positions and surprisal

| method | forks | mean relative position | median relative position | mean surprisal at fork token |
|---|---|---|---|---|
| EPTree (B2) | 82000 | 0.48 | 0.49 | 4.80 |
| Random (B1) | 18000 | 0.45 | 0.44 | 0.20 |
