# RL baselines: results and analysis

Status as of 2026-10-01, 07:20 server time. Three methods were trained from Qwen2.5-Math-1.5B-Instruct on the same filtered `train_30k` problems:
- **TreeRL:** EPTree (6,2,1,2) with tree advantages.
- **ChainRL:** 8 independent samples, using TreeRL's advantage code on chains (RLOO-style).
- **GRPO:** the same 8 samples, with DeepSeekMath's group-normalized advantage.

Each method was run twice:
- **Short runs:** 40 steps at lr 1.5e-6 (the paper's learning rate).
- **Long runs:** 150 steps at lr 5e-6.

Every step uses 8 prompts × 8 trained responses (64 samples). All settings are in [../../RL_CHANGES.md](../../RL_CHANGES.md) and [../../GRPO_CHANGES.md](../../GRPO_CHANGES.md); how to reproduce everything is in [../../RUN_LOG.md](../../RUN_LOG.md).

## Files in this folder

| File | Content |
|---|---|
| `eval.csv` | Raw evaluation rows (`solve_rate.py --summary`): model, benchmark, problems, n, temperature, mean accuracy, pass@n |
| `eval_table.csv` | The same as one row per (run, step, decoding), with columns per benchmark (from `tree_alloc/rl_report.py`) |
| `train_curves.csv` | Per run and step: training-batch accuracy, loss, grad norm, response length, entropy, timings |
| `runs.csv` | Per run: steps logged, peak GPU memory of our processes |
| `train_logs/*.jsonl` | Raw per-step training logs (`ckpt/<run>/train_log.jsonl`) |
| `gpu_mem/*.log` | Our GPU memory, sampled every minute during training (MiB) |
| `pipeline.log` | `scripts/run_baselines.sh`'s own log: every stage, attempt and GPU |

Regenerate the CSV tables with `python -m tree_alloc.rl_report results/rl` (from `mid_submission/`).

## Run status

| Run | Steps done | Train time | Peak GPU memory | Notes |
|---|---|---|---|---|
| TreeRL, 40 steps | 40 / 40 | 67 min | 18.5 GB | done |
| ChainRL, 40 steps | 40 / 40 | 35 min | 21.0 GB | done (the pipeline log's "not finished after 3 tries" is a logging bug, now fixed: the third attempt succeeded) |
| GRPO, 40 steps | 40 / 40 | 35 min | 18.7 GB | done |
| TreeRL, 150 steps | 150 / 150 | 156 min (1.0 min/step) | 20.7 GB | done |
| **ChainRL, 150 steps** | **64 / 150** | — | 18.7 GB | **stopped**: 3 attempts failed (see below); checkpoint at step 50 |
| **GRPO, 150 steps** | running (step 27 of attempt 3) | — | 18.7 GB | attempts 1–2 died before the first checkpoint, so attempt 3 started over |

ChainRL and GRPO train in about half TreeRL's time. With 8 responses per prompt, their rollouts take ~17 s per step against ~32 s for TreeRL's 30-leaf trees. So the comparison is matched on *trained* responses (8 per prompt). TreeRL *generates* ~2.4× more tokens per step: Task 2 measured ~14.5k tokens per (6,2,1,2) tree against ~5.9k for 8 independent chains.

**Why the long runs failed: other users on the shared GPU, not the code.**
- `CUDA Error: out of memory` in vLLM's allocator (ChainRL attempt 1, GRPO attempts 1–2). vLLM sleeps during each training step and releases its ~14 GB. Another user's job took that memory, and vLLM could not wake up.
- `No available memory for the cache blocks` (ChainRL attempt 3). The run restarted on GPU0 when it showed enough free memory, but another job grew while vLLM was starting.

## Evaluation results

Accuracy in %. Greedy = one answer per problem. mean@8 = average accuracy over 8 samples per problem at T = 1.0, the training temperature; pass@8 is in brackets. Benchmarks: MATH500 (500 problems), AMC (83) and Omni-MATH-500 (499, the Task 2 set).

### Short runs (40 steps, lr 1.5e-6)

| | MATH500 greedy | AMC greedy | Omni greedy | MATH500 mean@8 | AMC mean@8 | Omni mean@8 |
|---|---|---|---|---|---|---|
| Base model | 74.6 | 47.0 | 28.5 | 73.5 (87.6) | 46.2 (69.9) | 26.2 (40.3) |
| TreeRL @20 | 74.4 | 44.6 | 28.7 | 73.6 (88.0) | 46.4 (72.3) | 26.1 (40.1) |
| TreeRL @40 | 74.4 | 47.0 | 27.3 | 73.6 (88.6) | 44.4 (68.7) | 26.3 (40.7) |
| ChainRL @20 | 74.8 | 49.4 | 29.7 | 74.1 (90.2) | 44.1 (67.5) | 26.2 (39.5) |
| ChainRL @40 | 75.4 | 49.4 | 28.9 | 73.7 (89.0) | 43.2 (67.5) | 25.9 (39.5) |
| GRPO @20 | 74.8 | 48.2 | 28.7 | 74.1 (89.8) | 44.1 (71.1) | 26.4 (40.1) |
| GRPO @40 | 75.4 | 48.2 | 29.5 | 74.3 (88.8) | 45.5 (72.3) | 26.3 (40.7) |

Greedy scores at steps 10 and 30 are in `eval_table.csv`; they show the same picture.

### Long runs (150 steps, lr 5e-6), mean@8 (pass@8)

| | MATH500 | AMC | Omni |
|---|---|---|---|
| Base model | 73.5 (87.6) | 46.2 (69.9) | 26.2 (40.3) |
| TreeRL @50 | 73.9 (89.0) | 44.6 (67.5) | 26.5 (38.9) |
| TreeRL @100 | 74.3 (89.2) | 43.2 (71.1) | 27.2 (41.1) |
| **TreeRL @150** | **75.2 (89.6)** | **48.0 (72.3)** | 26.7 (39.7) |
| ChainRL @50 | 74.4 (88.6) | 45.5 (67.5) | 26.9 (40.5) |
| ChainRL @150, GRPO @50–150 | not available yet | | |

### How big is the noise?

One standard error (SE) for a single accuracy on these benchmark sizes is about **±2.0 points on MATH500 and Omni, and ±5.5 on AMC**. The mean@8 scores are only a little tighter, because most of the spread comes from *which* problems are in the set, not from sampling.

Comparing two models **on the same problems** (a paired test) cancels the problem effect. Its SE should be well under 1 point on MATH500. But that needs per-problem results, and the pipeline saved only totals (see rerun B).

## What the results say

1. **The short runs did not change the model.** At 40 steps × 64 samples and lr 1.5e-6, all 18 checkpoint evaluations (12 greedy, 6 mean@8), on all three benchmarks, are within about 1 SE of the base model, in both directions. The largest gaps are AMC: +2.4 for ChainRL greedy and −3.0 for ChainRL mean@8, against ±5.5 SE. The three methods cannot be ranked from these runs; at this budget they are equivalent.
2. **Long TreeRL is the first run that may have improved the model,** and only at the end: at step 150, MATH500 mean@8 is +1.7 (73.5 → 75.2), AMC +1.8, pass@8 +2.0 on MATH500; Omni is +0.5. Steps 50 and 100 are flat. That is consistent with a slow real gain, but each difference is under 1 unpaired SE, so it is **not established** without the paired test.
3. **Training metrics show no clear trend for any method.** Training-batch accuracy stays between 0.39 and 0.54 per 10–25-step window (`train_curves.csv`), but each step sees 8 new problems, so this curve is mostly problem-difficulty noise. Response length (~800–900 tokens) and entropy (0.11–0.20) stay stable: no collapse and no length blow-up, even at lr 5e-6.
4. **TreeRL vs. the baselines cannot be decided yet.** Long ChainRL has only step 50 (74.4 / 45.5 / 26.9, similar to TreeRL @50), and long GRPO has no checkpoint yet.
5. **Compute and memory.** All runs peaked at 18.5–21 GB of GPU memory, so they fit easily in 40 GB. On a GPU without other users, a TreeRL step takes ~1 min, and a ChainRL/GRPO step about half that.

## Update 2026-10-01 evening: paired results, data overlap, paper protocol

**Paired results** (`paired.csv`, 8 samples per problem, same problems as the base model, pooled over all 1,082 test problems):

| Checkpoint | Gain vs base | 95% CI |
|---|---|---|
| TreeRL long @50 | +0.20 | [−0.45, +0.87] |
| TreeRL long @100 | +0.23 | [−0.51, +0.94] |
| **TreeRL long @150** | **+0.92** | **[+0.20, +1.69]** |
| ChainRL long @50 | +0.12 | [−0.64, +0.84] |
| GRPO long @50 | −0.37 | [−1.04, +0.34] |

By benchmark, TreeRL @150 is MATH500 +0.95 [−0.25, +2.08], AMC +3.16 [−0.30, +6.63] and Omni +0.53 [−0.48, +1.58]. The earlier unpaired "+1.7 on MATH500" came from comparing two separate scorings: re-scoring the base model gave 74.0 instead of 73.5. AMC is now counted over all 83 problems; its ids repeat across its two contests, and an earlier version keyed on them and counted 49.

**Train/test overlap.** `train_30k` contains Omni-MATH problems, and 454 of the 500 Omni-MATH-500 test problems are in it. Our 1,200 long-run training problems contain 14 Omni-MATH-500 problems and 1 MATH500 problem; AMC is clean. With those 15 removed, TreeRL @150 is **+0.78 [+0.04, +1.57]** over the remaining 1,067 problems. Future training sets should drop every eval problem before sampling.

**How the paper sets up TreeRL vs. ChainRL (Sec. 4.1), next to ours:**
- **Paper:** TreeRL (6,2,1,2) generates 30 answers per problem and trains on **all 30** (batch 480). ChainRL samples **16** chains (batch 256). The two are matched on *generated tokens*.
- **Ours:** every method trains on 8 answers per problem; ChainRL and GRPO sample 8 chains. So our TreeRL generates ~2.4× the tokens of our baselines but trains on the same number of answers.
- **Other settings:** the paper uses KL β = 1e-4 (its released script uses 0, as we do), temperature 1.2 (we use 1.0) and max length 8,192 (we use 2,048).

The page "TreeRL Experiment Map" (claude.ai artifact) puts the flow, the code origin, the settings, the data and these results on one page.

## Reruns needed

| | What | Why | Cost | Priority |
|---|---|---|---|---|
| **A** | Finish long ChainRL (resume from step 50) and long GRPO (if attempt 3 fails too) | Needed for the only comparison that may show a difference (long TreeRL vs. long baselines) | ~1.5–2.5 h each | **Required** |
| **B** | Re-score the base model and every long-run checkpoint with per-problem output (`solve_rate.py --out`, 8 samples), then paired bootstrap CIs (as Task 2 did) | Without it, TreeRL @150's +1.7 on MATH500 can be neither claimed nor ruled out | ~10 min per checkpoint, ~1.5 h in total | **Required** to report any difference |
| C | Second seed for the three long runs | Measures run-to-run RL variance, which may exceed the method differences | ~8 h | Recommended if B shows an effect |
| D | Short (40-step) runs | Conclusively flat; more evaluation won't change that | — | **Not needed** |

**Status of A and B (queued 2026-10-01 07:55 server time, after the pipeline instance running long GRPO attempt 3).** `scripts/run_baselines.sh` now:
- resumes long ChainRL from step 50 (and long GRPO, if needed) with `KEEP_VLLM=1 VLLM_MEM=0.25`, the first option below. A 0.5B smoke step confirmed vLLM no longer sleeps.
- writes per-problem results for every evaluation (`mid_submission/results/rl_per_problem/`). It first re-scores the base model and long TreeRL's checkpoints, then every new long-run checkpoint. `rl_report.py` turns these into `paired.csv`: each checkpoint minus the base on the same problems, with a 95% bootstrap CI. It was tested on synthetic data (+4.6 points → CI [+4.1, +5.1]).

The options that were considered:

**Before A, make the runs robust to other users.** Two options:
- **Keep vLLM's memory between rollouts** (no sleep). Peak memory rises from ~19 GB to ~27 GB, but nobody can take vLLM's memory while it sleeps, which caused all the long-run failures so far. A small change: a switch that skips `sleep()`, and a higher `NEED_RL`.
- **Or checkpoint every 25 steps and allow more attempts.** Less progress is lost per failure, at ~19 GB more disk per run. `/home` has 89 GB free and is 95% full.

Also, a resumed run restarts Adam's moment estimates; the learning rate is unaffected (constant). This applies to the long ChainRL run past step 50.
