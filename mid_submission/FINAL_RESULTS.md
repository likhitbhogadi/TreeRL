# Final results: TreeRL vs ChainRL vs our method (with three ablations)

All runs fine-tune **Qwen2.5-Math-1.5B-Instruct** with RL at the TreeRL paper's settings (scaled to one GPU), and are
scored on the same 1,755 test problems. Every difference is reported with a paired 95% bootstrap confidence interval.

**Bottom line.** On greedy accuracy, our method matches or slightly beats TreeRL. At step 150, our main run has the best
score of any run (48.55, +0.34 over TreeRL at step 150, +0.68 over TreeRL at its final step 200). At step 100, all four
of our configs are ahead of TreeRL (+0.28 to +0.74 points) and ChainRL (+0.57 to +1.03). None of these greedy gaps is
statistically significant. With 8 sampled answers, every method stays within ±0.6 points of the starting model.
Starting from an already RL-trained model, no method in this budget moves accuracy beyond noise.

---

## 1. Data

### Training set

`mid_submission/data/train_30k_mixed_12k.jsonl` holds **5,504 math problems**, the same file for every run (checksum
`403789637`, recorded in each run's `run_config`).

How it was built (`solve_rate.py`, commands in [RERUN_PAPER.md](RERUN_PAPER.md)):
1. **Sample:** 12,000 problems from `datasets/train/train_30k.jsonl`.
2. **Measure difficulty:** the starting model answers each problem 8 times (temperature 1.0, top-p 0.95, up to 3,072
   tokens).
3. **Keep "mixed" problems:** only those solved 1 to 7 times out of 8. Problems that are always or never solved give
   every response the same reward, so they carry no learning signal.
4. **Remove test overlap:** drop any problem whose text matches a test problem (ignoring whitespace and case).
   `train_30k` contained 454 of the 500 Omni-MATH-500 test problems and 98 of the OlympiadBench ones.

The starting model's mean solve rate on this set is about 0.45. Each training step uses 16 problems, so 200 steps use
3,200 problems and 150 steps use 2,400. Every run uses seed 42.

### Test sets

| Benchmark | File | Problems |
|---|---|---|
| MATH500 | `datasets/eval/MATH500.jsonl` | 500 |
| AMC | `datasets/eval/aimo-validation-amc.jsonl` | 83 |
| Omni-MATH-500 | `mid_submission/data/omni_math_500_seed0.jsonl` | 499 |
| OlympiadBench | `datasets/eval/olympiad_bench.jsonl` | 673 |
| **Total** | | **1,755** |

These are 4 of the paper's 6 benchmarks. AIME2024 (30 problems) is not in the repo, and LiveCodeBench is a code
benchmark, which doesn't fit a math-only model and dataset.

---

## 2. Evaluation metrics

Every checkpoint at steps 50, 100, 150 (and 200 for the baselines; 125 for ablation 3) is scored by `solve_rate.py`
with vLLM, with a 3,072-token answer limit.

| Metric | How it's computed | Notes |
|---|---|---|
| **Greedy accuracy** (main metric) | 1 answer per problem at temperature 0; % of problems correct | The paper's metric. No randomness. Noise is about ±1 point. |
| **Mean@8** | 8 answers per problem at temperature 1.0, top-p 0.95; % correct averaged over all 8 | Accuracy under sampling, as during training. Noise is about ±0.5 point. |
| **Pass@8** | % of problems where at least 1 of the 8 sampled answers is correct | Measures coverage, not reliability. |

- **Answer checking:** the last `\boxed{...}` in the response is compared with the reference answer using
  `math_verify` (`tree_alloc/verify.py`). A response cut off at 3,072 tokens counts as wrong.
- **Pooled score:** the average over all 1,755 problems, so each problem counts once. Larger benchmarks weigh more.
- **Confidence intervals:** paired bootstrap (`tree_alloc.metrics.paired_bootstrap`, 2,000 resamples). It resamples
  test problems, and both models are compared on the same problems, so problem difficulty cancels out. The CI is the
  2.5th–97.5th percentile of the mean difference. If the CI contains 0, the difference is within noise.
- **Comparison script:** [final_ci.py](final_ci.py) (run from `mid_submission/`) reads the per-problem files in
  `results/rl_paper_per_problem/` (mean@8) and `results/rl_paper_per_problem_greedy/` (greedy).

---

## 3. Experiment configurations

### Shared settings (all six runs)

| Setting | Value | vs the TreeRL paper |
|---|---|---|
| Model | Qwen2.5-Math-1.5B-Instruct | Paper: Qwen2.5-14B SFT |
| Prompts per step | 16 | Same |
| Learning rate | 1.5e-6, 10-step warmup, then constant | Same |
| KL penalty (β) | 1e-4, per-token reward penalty | Same as the paper's text |
| Optimizer steps per rollout batch | 1 | Same |
| Sampling (training) | temperature 1.0, top-p 0.95 | Paper: 1.2 / 0.95 (1.2 garbled 13% of this model's answers) |
| Max response length | 3,072 tokens | Paper: 8,192 (our model's context is 4,096) |
| Reward | +1 correct, 0 wrong; −1 on every token of a truncated response | Same reward; truncation penalty from the released code |
| Advantage | (global + local advantage) / √n, the released TreeRL code | Same |
| Leaves trained | All responses generated | Same |
| Seed | 42 (one seed per config) | Paper doesn't state |

### Methods

The tree shape is written (M, N, L, T): M initial chains, then L rounds of forking, where each round picks N
high-entropy tokens per chain and generates T new branches from each. One tree gives **M(1 + N·L·T)** responses.

| Run | Rollout per prompt | Responses / prompt | Trained per step | Steps |
|---|---|---|---|---|
| **TreeRL** | EPTree (6,2,1,2) | 30 | 480 | 200 |
| **ChainRL** | 16 independent chains | 16 | 256 | 200 |
| **Ours, main** | Phase I EPTree (4,2,1,2) → 20, then Phase II k = 10; h = 8, λ = 0.5 | 30 | 480 | 150 |
| **Ablation 1** | Phase I EPTree (2,2,1,2) → 10, then Phase II k = 20; h = 8, λ = 0.5 | 30 | 480 | 150 |
| **Ablation 2** | Phase I EPTree (4,1,1,3) → 16, then Phase II k = 14; h = 20, λ = 0.5 | 30 | 480 | 150 |
| **Ablation 3** | Phase I EPTree (4,2,1,2) → 20, then Phase II k = 10; h = 8, λ = 0.3 | 30 | 480 | 125* |

\* Ablation 3 was scheduled for 150 steps but stopped at step 125 when its compute credit ran out, so it's scored at
steps 50, 100 and 125.

**Our method (Phase II), in brief** (`openrlhf/trainer/ppo_utils/credit_aware_tree.py`, `adithya-rl` branch). After
Phase I builds a smaller EPTree, the remaining k rollouts are spent one at a time. Each rollout branches from the
state v* that maximises

> J(v) = λ · Ñ(parent(v)) + (1 − λ) · C̃(v)

- **Novelty** N_h(u) is the probability that a new rollout from u differs from every continuation already seen there
  within h tokens.
- **Credit** C(v) = |V(v) − V(parent(v))| is how much the mean reward changes when the path steps into v.
- Ñ and C̃ are min-max normalised over the current candidates, and the tree is updated after every rollout.
- λ weights novelty and 1 − λ weights credit, so λ = 0.3 puts 70% of the weight on credit.

Every config of ours uses 30 responses per prompt, the same budget as TreeRL.

**What the ablations change** (each compared with the main run):
- **Ablation 1:** only the Phase I / Phase II split, shifting budget to Phase II (10 + 20 instead of 20 + 10).
- **Ablation 2:** the split (16 + 14) and the novelty horizon (h = 20 instead of 8). It can't separate the two effects.
- **Ablation 3:** only λ (0.3 instead of 0.5), putting more weight on credit than novelty.

### Hardware and speed

| Run | GPU | Adam | Minutes / step |
|---|---|---|---|
| TreeRL | RTX PRO 6000 96 GB (JarvisLabs) | GPU | 1.6 |
| ChainRL | RTX PRO 6000 96 GB (JarvisLabs) | GPU | 0.9 |
| Ours, main | RTX PRO 6000 96 GB (JarvisLabs VM) | GPU | 2.6 |
| Ablation 1 | L40S 48 GB (pkgpu2) | CPU (offloaded) | 7.6 |
| Ablation 2 | L40S 48 GB (pkgpu2) | CPU (offloaded) | 6.8 |
| Ablation 3 | RTX PRO 6000 96 GB (JarvisLabs VM) | GPU | 2.6 |

On the same GPU, our method takes about 1.6× as long per step as TreeRL (2.6 vs 1.6 min). Phase II generates its
rollouts one after another and recomputes the tree's scores after each one.

**Resumes.** Two runs were resumed from a saved checkpoint with a fresh optimizer state (weights only; the KL reference
stays the starting model):
- **Ablation 1** was killed several times when pkgpu2 ran out of host RAM (other users' jobs) and resumed from step 75.
  Its step-100 and step-150 checkpoints come after the resume.
- **Ablation 2** stopped at step 102 for time and later resumed from step 100. Its step-150 checkpoint comes after
  the resume.

All other runs trained straight through to every scored checkpoint.

---

## 4. Results

### 4.1 Greedy accuracy, all 1,755 problems (main metric)

| Step | TreeRL | ChainRL | Ours main | Ablation 1 | Ablation 2 | Ablation 3 |
|---|---|---|---|---|---|---|
| 0 (starting model) | 47.52 | 47.52 | 47.52 | 47.52 | 47.52 | 47.52 |
| 50 | **47.69** | 47.29 | 47.07 | 47.46 | 47.46 | 47.24 |
| 100 | 47.07 | 46.78 | 47.35 | 47.58 | **47.81** | 47.52 |
| 125 | — | — | — | — | — | 47.75 |
| 150 | 48.21 | 47.52 | **48.55** | 47.64 | 47.46 | — |
| 200 | 47.86 | 47.69 | — | — | — | — |

### 4.2 Paired differences vs the baselines, greedy (95% CI)

| Step | Run | vs starting model | vs TreeRL (same step) | vs ChainRL (same step) |
|---|---|---|---|---|
| 50 | Ours main | −0.46 [−1.31, +0.40] | −0.63 [−1.65, +0.28] | −0.23 [−1.08, +0.63] |
| 50 | Ablation 1 | −0.06 [−1.03, +0.91] | −0.23 [−1.31, +0.85] | +0.17 [−0.85, +1.14] |
| 50 | Ablation 2 | −0.06 [−1.08, +0.97] | −0.23 [−1.14, +0.68] | +0.17 [−0.85, +1.25] |
| 50 | Ablation 3 | −0.28 [−1.31, +0.80] | −0.46 [−1.48, +0.57] | −0.06 [−1.03, +1.03] |
| 100 | Ours main | −0.17 [−1.25, +0.85] | +0.28 [−0.80, +1.37] | +0.57 [−0.46, +1.60] |
| 100 | Ablation 1 | +0.06 [−0.91, +0.97] | +0.51 [−0.63, +1.65] | +0.80 [−0.23, +1.88] |
| 100 | Ablation 2 | +0.28 [−0.80, +1.37] | **+0.74** [−0.40, +1.77] | **+1.03** [−0.11, +2.11] |
| 100 | Ablation 3 | +0.00 [−1.14, +1.08] | +0.46 [−0.68, +1.60] | +0.74 [−0.40, +1.82] |
| 125 | Ablation 3 | +0.23 [−0.85, +1.37] | −0.46 [−1.71, +0.80] (TreeRL @150) | +0.23 [−1.08, +1.48] (ChainRL @150) |
| 150 | Ours main | **+1.03** [−0.17, +2.17] | +0.34 [−0.80, +1.48] | **+1.03** [−0.17, +2.11] |
| 150 | Ablation 1 | +0.11 [−0.97, +1.20] | −0.57 [−1.77, +0.63] | +0.11 [−1.03, +1.25] |
| 150 | Ablation 2 | −0.06 [−1.20, +1.03] | −0.74 [−1.82, +0.34] | −0.06 [−1.25, +1.08] |

For reference: TreeRL vs the starting model is +0.17 / −0.46 / +0.68 at steps 50 / 100 / 150, and ChainRL is
−0.23 / −0.74 / +0.00. Ours main at step 150 vs TreeRL at its final step 200: +0.68 [−0.46, +1.82].

### 4.3 Ablations vs the main run (95% CI)

| Ablation | Step | Greedy | Mean@8 |
|---|---|---|---|
| 1 (split 10 + 20) | 50 | +0.40 [−0.46, +1.31] | −0.24 [−0.66, +0.17] |
| 1 | 100 | +0.23 [−0.74, +1.25] | −0.28 [−0.75, +0.21] |
| 1 | 150 | −0.91 [−2.11, +0.34] | **+0.51 [+0.06, +0.99]** |
| 2 (split 16 + 14, h = 20) | 50 | +0.40 [−0.57, +1.37] | +0.09 [−0.33, +0.48] |
| 2 | 100 | +0.46 [−0.63, +1.48] | +0.14 [−0.35, +0.66] |
| 2 | 150 | −1.08 [−2.22, +0.00] | +0.39 [−0.09, +0.87] |
| 3 (λ = 0.3) | 50 | +0.17 [−0.85, +1.20] | +0.07 [−0.37, +0.51] |
| 3 | 100 | +0.17 [−0.85, +1.20] | −0.35 [−0.82, +0.14] |
| 3 | 125 (vs main @150) | −0.80 [−2.05, +0.46] | +0.41 [−0.06, +0.90] |

### 4.4 Greedy accuracy per benchmark

| Run | MATH500 | AMC | Omni-MATH | OlympiadBench | Pooled |
|---|---|---|---|---|---|
| Starting model | 74.4 | 47.0 | 29.1 | 41.3 | 47.52 |
| TreeRL @100 | 74.4 | 47.0 | 26.9 | 41.8 | 47.07 |
| ChainRL @100 | 74.2 | 43.4 | 27.7 | 41.0 | 46.78 |
| Ours main @100 | 73.4 | 48.2 | 29.1 | 41.5 | 47.35 |
| Ablation 1 @100 | 75.4 | 50.6 | 28.5 | 40.7 | 47.58 |
| Ablation 2 @100 | 74.8 | 47.0 | 28.9 | 41.9 | 47.81 |
| Ablation 3 @100 | 76.6 | 45.8 | 27.3 | 41.2 | 47.52 |
| Ablation 3 @125 | 75.2 | 47.0 | 29.1 | 41.3 | 47.75 |
| TreeRL @150 | 76.0 | 48.2 | 28.1 | 42.5 | 48.21 |
| ChainRL @150 | 74.6 | 48.2 | 27.5 | 42.2 | 47.52 |
| Ours main @150 | 76.6 | 50.6 | 28.3 | 42.5 | 48.55 |
| Ablation 1 @150 | 75.6 | 48.2 | 29.7 | 40.1 | 47.64 |
| Ablation 2 @150 | 76.2 | 49.4 | 27.5 | 40.7 | 47.46 |

AMC has only 83 problems, so one problem is 1.2 points.

### 4.5 Sampled answers: mean@8 and pass@8, all 1,755 problems

| Step | TreeRL | ChainRL | Ours main | Ablation 1 | Ablation 2 | Ablation 3 |
|---|---|---|---|---|---|---|
| 0 | 45.46 / 62.56 | 45.46 / 62.56 | 45.46 / 62.56 | 45.46 / 62.56 | 45.46 / 62.56 | 45.46 / 62.56 |
| 50 | 45.44 / 62.68 | 45.40 / 62.79 | 45.26 / 62.91 | 45.02 / 62.56 | 45.34 / 63.08 | 45.33 / 63.13 |
| 100 | 45.15 / 62.45 | 45.32 / 62.79 | 45.23 / 62.34 | 44.95 / 61.14 | 45.37 / 62.56 | 44.88 / 62.51 |
| 125 | — | — | — | — | — | 45.41 / 63.42 |
| 150 | 44.99 / 62.22 | 45.39 / 62.74 | 45.00 / 62.74 | 45.51 / 62.85 | 45.39 / 62.68 | — |
| 200 | 45.02 / 62.45 | 45.09 / 62.68 | — | — | — | — |

Each cell is mean@8 / pass@8.

Mean@8 differences that are significant (CI excludes 0), against the baselines and the starting model:
- **Ablation 1 @150 vs TreeRL @150: +0.52 [+0.02, +1.03]** (better)
- Ablation 1 @50 vs TreeRL @50: −0.42 [−0.84, −0.01] (worse)
- Ablation 1 @100 vs starting model: −0.51 [−0.98, −0.04] (worse)
- Ablation 3 @100 vs starting model: −0.58 [−1.07, −0.09] (worse)

Against the main run, only ablation 1 @150 is significant (+0.51, §4.3). All other mean@8 differences have CIs that
include 0. Ablation 2 @150 vs TreeRL @150 is +0.40 [−0.11, +0.88], and ablation 3 @125 vs TreeRL @150 is +0.42
[−0.06, +0.92].

### 4.6 Training-batch accuracy (fraction of training responses correct, averaged over 25-step windows)

| Steps | TreeRL | ChainRL | Ours main | Ablation 1 | Ablation 2 |
|---|---|---|---|---|---|
| 1–25 | 0.437 | 0.441 | 0.434 | 0.429 | 0.425 |
| 26–50 | 0.458 | 0.460 | 0.426 | 0.430 | 0.451 |
| 51–75 | 0.441 | 0.453 | 0.446 | 0.462 | 0.450 |
| 76–100 | 0.435 | 0.443 | 0.403 | 0.417 | 0.439 |
| 101–125 | 0.444 | 0.443 | 0.453 | 0.456 | 0.434 |
| 126–150 | 0.444 | 0.457 | 0.454 | 0.453 | 0.443 |
| 151–175 | 0.439 | 0.446 | — | — | — |
| 176–200 | 0.416 | 0.422 | — | — | — |

Ablation 3's training log is on the paused JarvisLabs VM and wasn't copied, so it isn't in this table.

Mean response length stayed at 830–880 tokens for every run throughout. For our method, Phase II picks where to branch,
so its batch accuracy isn't directly comparable with TreeRL's or ChainRL's.

Raw files: [results/rl_paper_eval.csv](results/rl_paper_eval.csv) has per-benchmark accuracy and pass@8 for every
checkpoint; per-problem results are in [results/rl_paper_per_problem/](results/rl_paper_per_problem/) and
[results/rl_paper_per_problem_greedy/](results/rl_paper_per_problem_greedy/).

---

## 5. Analysis

**1. Our method is competitive with TreeRL at equal budget, but the gap is within noise.** Our configs use the same 30
responses per prompt and 480 trained responses per step as TreeRL.
- At step 150, the main run has the highest greedy score of any run or step (48.55), ahead of TreeRL at both step 150
  (+0.34) and its final step 200 (+0.68). It is also best or tied-best on MATH500 (76.6), AMC (50.6) and OlympiadBench
  (42.5).
- At step 100, all four of our configs are ahead of TreeRL on greedy accuracy (+0.28 to +0.74) and of ChainRL (+0.57
  to +1.03).

Every one of these CIs includes 0, so the accurate claim is "matches or slightly exceeds TreeRL", not "beats". The
consistent direction at step 100 (4 of 4 configs ahead of both baselines) is suggestive but comes from one seed per
config.

**2. No method improves the starting model beyond noise.** The largest gain over the starting model is +1.03 greedy
(ours main @150, CI [−0.17, +2.17]). Neither TreeRL nor ChainRL moved it by more than 0.7 at any step. Mean@8 and
pass@8 are flat for every method, and training-batch accuracy stays at 0.40–0.46 throughout. Likely reasons:
- **The starting model has already been through RL.** Qwen trained Qwen2.5-Math-1.5B-Instruct with GRPO; the paper
  starts from SFT models. Even the paper's strongest starting model (R1-distilled 7B) gained only +0.9 (ChainRL) and
  +2.2 (TreeRL).
- **Scale.** 150–200 steps with 3,072-token answers is well below the paper's ~300 steps with 8,192 tokens.

We also tried the base model (Qwen2.5-Math-1.5B), which has more room to improve. It produced long looping answers (17%
hit the length limit), which made training about 3× slower, so it couldn't finish in time and was dropped.

**3. Greedy and sampled metrics disagree.** Our main run's gains appear only in greedy accuracy; its mean@8 is flat
(45.00 at step 150, vs 44.99 for TreeRL). RL here mostly changes which answer the model picks first, not how often
sampled answers are right.

**4. Ablations.** With one seed each and almost every difference within noise, the ablations don't identify a better
config than the main run. The weak patterns:
- **The main run starts slowest and finishes best.** It is the lowest of our configs at step 50 (47.07), and every
  ablation is ahead of it at steps 50 and 100 (by +0.17 to +0.46). By step 150 it leads ablations 1 and 2 by about one
  point (−0.91 and −1.08 for the ablations), and ablation 3 at step 125 is also below it (−0.80).
- **Ablation 1 (more Phase II budget, k = 20)** was flat on greedy (47.46 → 47.64). On mean@8 it is the only config
  significantly above the main run (@150, +0.51) and above TreeRL (+0.52), but it also has two results significantly
  below a baseline. Its later checkpoints come after a resume.
- **Ablation 2 (h = 20, split 16 + 14)** was the best run at step 100 (47.81) but fell to 47.46 by step 150, the
  largest gap to the main run (−1.08, CI upper end at 0.00). Its steps 101–150 came after a resume with a fresh
  optimizer state, which may have contributed.
- **Ablation 3 (λ = 0.3, more weight on credit)** tracked the main run closely through step 100 (+0.17 at both 50 and
  100). Its mean@8 dipped at step 100 (significantly below the starting model, −0.58) and recovered by step 125
  (45.41). It stopped at step 125, so it can't be compared at step 150.

**5. Multiple comparisons.** About 60 comparisons appear in this report. At a 95% level, about three would cross 0 by
chance even if all methods were identical. The five "significant" mean@8 results (§4.3 and §4.5; two up, three down)
are consistent with that and shouldn't be read as real effects.

**6. Cost.** On the same GPU, our method is about 1.6× slower per step than TreeRL (2.6 vs 1.6 min). Phase II's k
rollouts are generated one at a time, and its scores are recomputed after each one. Batching Phase II (several branch
points per round) would recover most of that.

---

## 6. Limitations

- **One seed per config.** Seed-to-seed variation is unknown and could be as large as the gaps between methods.
- **Unequal steps.** Our runs stop at step 150 (ablation 3 at 125) versus 200 for the baselines. Comparisons are made
  at matching steps.
- **Ablation 2 changes two factors at once** (h and the split).
- **Different hardware.** Different GPUs and Adam placement (GPU vs CPU offload) for different runs. This changes
  speed; numerical effects on results should be negligible but weren't measured.
- **Resumes.** Ablations 1 and 2 resumed from a checkpoint with a fresh optimizer state (§3).
- **Ablation 3 stopped at step 125** (compute credit ran out). If steps 125–150 repeated its step 100–125 change, step
  150 would be about 47.98 greedy and 45.94 mean@8. This is an extrapolation, not a measurement, so it isn't in the
  tables.
- **Scaled replication.** 4 of the paper's 6 benchmarks, a 1.5B model, shorter answers and fewer steps than the paper.
