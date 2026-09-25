# Task 2 results: EPTree replication

243 problems common to all runs; i.i.d. baseline from `b0_64.jsonl` (64 chains/problem).

## i.i.d. chains (B0), unbiased pass@k

| k | tokens/problem | PassRate | distinct answers |
|---|---|---|---|
| 1 | 812 | 45.8% | 1.00 |
| 2 | 1637 | 63.0% | 1.62 |
| 4 | 3316 | 77.8% | 2.66 |
| 8 | 6559 | 88.5% | 4.09 |
| 16 | 13096 | 95.0% | 6.13 |
| 32 | 26173 | 98.2% | 9.14 |
| 64 | 52489 | 99.6% | 13.31 |

## Tree configs (Fig. 4 / Fig. 5 data)

| method | (M,N,L,T) | leaves | tokens/problem | PassRate | i.i.d. PassRate at same tokens | mean acc | distinct answers | mixed-outcome trees |
|---|---|---|---|---|---|---|---|---|
| EPTree (B2) | (4,1,3,1) | 16 | 8773 | 90.1% | 91.3% | 0.443 | 5.65 | 87% |
| Random (B1) | (4,3,1,1) | 16 | 9024 | 92.2% | 91.5% | 0.463 | 5.47 | 86% |
| EPTree (B2) | (4,3,1,1) | 16 | 9454 | 93.8% | 92.0% | 0.438 | 5.86 | 90% |
| EPTree (B2) | (8,1,1,1) | 16 | 10772 | 94.2% | 93.2% | 0.444 | 5.95 | 92% |
| Random (B1) | (6,2,1,2) | 30 | 16377 | 97.9% | 96.1% | 0.451 | 8.05 | 97% |
| EPTree (B2) | (6,2,1,2) | 30 | 17370 | 97.9% | 96.3% | 0.451 | 8.70 | 97% |
| EPTree (B2) | (8,1,7,1) | 64 | 29444 | 99.6% | 98.5% | 0.421 | 12.93 | 99% |
| EPTree (B2) | (8,7,1,1) | 64 | 36000 | 98.4% | 98.8% | 0.447 | 13.05 | 98% |
| Random (B1) | (16,3,1,1) | 64 | 36141 | 99.2% | 98.9% | 0.457 | 12.80 | 99% |
| EPTree (B2) | (16,3,1,1) | 64 | 38201 | 100.0% | 99.0% | 0.440 | 13.56 | 100% |

## Table 2: entropy-guided vs. random forking (same M,N,L,T)

| (M,N,L,T) | PassRate B2 | PassRate B1 | Δ PassRate [95% CI] | acc B2 | acc B1 | distinct B2 | distinct B1 | tokens B2 | tokens B1 |
|---|---|---|---|---|---|---|---|---|---|
| (4,3,1,1) | 93.8% | 92.2% | +1.6 [-2.1, +5.3] | 0.438 | 0.463 | 5.86 | 5.47 | 9454 | 9024 |
| (6,2,1,2) | 97.9% | 97.9% | +0.0 [-2.5, +2.1] | 0.451 | 0.451 | 8.70 | 8.05 | 17370 | 16377 |
| (16,3,1,1) | 100.0% | 99.2% | +0.8 [+0.0, +2.1] | 0.440 | 0.457 | 13.56 | 12.80 | 38201 | 36141 |

## Fig. 7: top-10 forking tokens (EPTree, all configs pooled)

| rank | token | share of forks |
|---|---|---|
| 1 | `␣\(` | 2.4% |
| 2 | `␣the` | 2.2% |
| 3 | `␣\` | 1.9% |
| 4 | `,` | 1.2% |
| 5 | `␣` | 1.0% |
| 6 | `1` | 0.9% |
| 7 | `␣we` | 0.9% |
| 8 | `␣and` | 0.8% |
| 9 | `.` | 0.8% |
| 10 | `:\n` | 0.8% |

## Fig. 8: fork positions and surprisal

| method | forks | mean relative position | median relative position | mean surprisal at fork token |
|---|---|---|---|---|
| EPTree (B2) | 52488 | 0.43 | 0.42 | 3.75 |
| Random (B1) | 20412 | 0.45 | 0.45 | 0.14 |
