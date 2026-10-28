# RL reruns with the paper's settings

The earlier RL runs (TreeRL, ChainRL, GRPO; [results/rl/ANALYSIS.md](results/rl/ANALYSIS.md)) used a reduced setup. This page covers why those runs barely moved the model, what changed in the code (including new logging), which settings and model the reruns use, the GPU plan for the ₹6,000 cloud budget, and how to run everything. The pipeline is [`scripts/run_paper.sh`](../scripts/run_paper.sh).

## 1. Why the earlier runs barely moved

**What happened.** Long runs: 150 steps, 8 prompts × 8 trained responses, lr 5e-6, scored on 1,082 test problems with paired CIs vs the base model:
- TreeRL @150: +0.92 [+0.17, +1.66].
- ChainRL @150: −0.16 [−0.85, +0.52].
- GRPO: −0.37 [−1.07, +0.30] @50, +0.05 [−0.69, +0.74] @100.

ChainRL and GRPO did not hurt the model; their small negative numbers are noise. Nothing moved by more than about 1 point.

**Causes, most important first.**

1. **About 15× less training than the paper.** We trained on 150 × 64 = 9,600 responses. The paper used ~300 steps × 480 = 144,000 for TreeRL and ~300 × 256 = 77,000 for ChainRL. In the paper the two methods are equal until ~100 of its steps (48,000 responses for TreeRL); our runs ended well before that.
2. **TreeRL trained on 8 of its 30 leaves; ChainRL and GRPO used 8 chains.** The paper trains TreeRL on all 30 leaves (its Table 3: 16 of 30 already costs 3.2 points) and gives ChainRL 16 chains, which match the tree's generated tokens.
   - The subset also biased the advantages. When it draws a subset, the released code (`tree_node.select_terminal`) always includes one correct leaf.
   - Running the released advantage code on simulated (6,2,1,2) trees: training on 8 of 30 leaves gives every response a mean advantage of +0.05 to +0.065. On all 30 leaves the mean is 0.000 ± 0.002, and for chains it is exactly 0.
   - Our TreeRL logs match the 8-leaf number: the mean per-token reward was +0.04 to +0.10, where ChainRL and GRPO logged 0.
   - A positive mean advantage acts like a little supervised fine-tuning on the model's own samples, which sharpens the policy. TreeRL's sampling entropy fell (0.15 → 0.13), while ChainRL's and GRPO's rose. So part of TreeRL's +0.9 may come from this bias rather than from tree credit assignment.
3. **lr 5e-6 with 64-response batches.** Adam scales every step to about the learning rate whether the gradient is signal or noise, and 64 responses give a noisy gradient. ChainRL's sampling entropy rose steadily over the run (0.15 → 0.28) and GRPO's too (0.14 → 0.22): the policy was drifting. The paper uses lr 1.5e-6 with batches of 256–480.
4. **ChainRL and GRPO were resumed from checkpoints; TreeRL never was.** ChainRL's long run resumed from a checkpoint and trained on 5 times (steps 50 and 100), GRPO's once (step 50), each time with a fresh optimizer state (`train_logs/`). TreeRL's ran straight through. This confounds the comparison.
5. **The starting model has already been through RL.** Qwen2.5-Math-1.5B-Instruct was trained with GRPO by Qwen, while the paper starts RL from SFT models. The paper's own already-strong model (R1-distilled 7B) gained only +0.9 (ChainRL) and +2.2 (TreeRL) average. Expect small gains here too, and rely on the paired CIs.

**Probably not a cause: KL β = 0.** In this code the KL term is added to each token's reward as −β × log(π/π_ref), with the log ratio clamped to ±0.5. At the paper's β = 1e-4 it shifts a token's advantage by at most 0.00005, against advantages of 0.2–2: a negligible difference, though not an identical objective. The reruns use the paper's 1e-4 anyway (§3). It loads a reference model: +3 GB, about +7% per step.

**Checked and correct:**
- tree → per-token advantage mapping;
- prompt/response masks and log-prob alignment;
- one optimizer step per rollout (the PPO ratio is 1);
- the GRPO advantage;
- weight sync (earlier check).


## 2. Code changes

| File | Change | Why |
|---|---|---|
| `scripts/treerl-qwen1.5b-1gpu.sh` | Defaults are now the paper's: `NUM_TRACE` = all leaves of the tree (M(1+NLT): 30 for (6,2,1,2), 16 for `"16 0 0 0"`). When every leaf is used it passes `--use_all_terminals` (the released code's flag for this). New knobs: `KL` (default 1e-4, the paper's), `SEED` (passed as `--seed`, default 42) and `ADAM_OFFLOAD` (empty = Adam on the GPU). It refuses three settings: `NUM_TRACE` > leaves, an `INFER_BS` that doesn't divide `NUM_TRACE`, and a data file shorter than `STEPS × ROLLOUT`. | Cause 2. Each refused setting silently goes wrong otherwise:<br>• an `INFER_BS` that doesn't divide `NUM_TRACE` drops the remainder of every tree from the log-prob pass;<br>• a short data file stops the trainer when it runs out of prompts (the 1,432-problem file would end a 16-prompt run at step 89).<br>Old commands that pass `NUM_TRACE=8` behave as before. |
| `train_reinforce_ray.py` | The KL reference model is built from the starting model, also after a resume. | It was built from `args.pretrain`, which `--resume` replaces with the newest checkpoint, so the reference moved with every interruption (KL > 0 only). |
| `openrlhf/trainer/ray/reinforce_actor.py` | The step count behind the 5% LR warmup is now counted in optimizer steps. | It divided prompt batches by micro-batches: 150 // 64 = 2 steps for our long runs, so warmup was one step at lr 0, and 0 steps at batch 480. Now 200 steps → 10 warmup steps, then constant lr (as before). After a resume (fresh Adam) the warmup also runs again. |
| `entropy_chain_local_manager.py`, `entropy_guided_tree_search.py`, `experience_maker.py`, `experience_maker_reinforce.py`, `reinforce_trainer.py` | New tree log `trees.jsonl` and new per-step fields in `train_log.jsonl` (§5), including `resumed_from`. Logging only: every logging step is wrapped so that it cannot stop a run. | The analyses need tree structure, credit and generated-token counts, which were not saved. |
| `mid_submission/solve_rate.py` | New `--exclude FILE...` drops training problems whose text (ignoring whitespace and case) appears in the given files. Per-problem output also gets mean response length and truncations. | `train_30k` contains 454 of the 500 Omni-MATH-500 test problems and 98 of the 675 OlympiadBench ones. 15 of them were in our old training set. |
| `mid_submission/tree_alloc/rl_report.py` | Reads OlympiadBench and the new train-log fields. | Earlier results regenerate unchanged (new columns stay empty). |
| `scripts/run_paper.sh` (new) | Unattended pipeline:<br>• waits for GPU memory, trains, resumes on failure;<br>• evaluates checkpoints greedy and with 8 samples per problem on MATH500, AMC, Omni-MATH-500 and OlympiadBench (1,757 problems).<br>`GPU=` pins an instance to one GPU, so two can run side by side:<br>• each run starts its own Ray cluster (`RAY_ADDRESS=local` in the launcher);<br>• an evaluation both instances need (the base model) runs once, under a lock.<br>Run names end in `-seed<SEED>`. Each run folder stores its settings in `run_config`; a folder with other settings is never resumed, reused or scored. Stage markers live in `<CKPT>/.done_paper`. It ends with `PAPER_RUNS_DONE`, or with `PAPER_RUNS_INCOMPLETE`, the unfinished stages and exit status 1; resumed runs are named. | §7 |
| `scripts/setup_cloud.sh`, `scripts/smoke_test.sh`, `scripts/smoke_check.py` (new) | One-time cloud setup (pkgpu2's package versions, model, Adam builds, an env file with the GPU's settings). A smoke test with automatic PASS/FAIL checks for all three methods and the evaluation; `BENCH=1` adds the timing run. | §7 |
| `scripts/run_baselines.sh` | Passes `KL=0`. | So it still reproduces the earlier runs now that the launcher defaults to 1e-4. |

The algorithm code is unchanged (`tree_node.py`, `gather_paths`, the loss).

## 3. Settings: a scaled replication

The reruns are a *scaled replication* of the paper's TreeRL vs ChainRL experiment, and the report should call them that.
- **Follows the paper:** tree shape, responses per prompt, prompt batch, learning rate, KL coefficient, top-p, reward and advantage.
- **Scaled to our model and budget:** model, temperature, response length, training data, number of steps and benchmarks.

| | Paper (Sec. 4.1, Table 3) | Earlier long runs | Reruns |
|---|---|---|---|
| Prompts per step | 16 | 8 | **16** |
| TreeRL tree / leaves trained | (6,2,1,2) / all 30 → batch 480 | (6,2,1,2) / 8 of 30 → 64 | **(6,2,1,2) / all 30 → 480** |
| ChainRL / GRPO chains | 16 → batch 256 | 8 → 64 | **16 → 256** |
| Optimizer steps per rollout | 1 | 1 | 1 |
| Learning rate | 1.5e-6 | 5e-6 | **1.5e-6**, 10-step warmup, then constant |
| KL β | 1e-4 (released script: 0) | 0 | **1e-4** (as the paper's text) |
| Steps | ~300 (Fig. 6) | 150 | **200**; 300 if the budget allows (§6) |
| Temperature / top-p | 1.2 / 0.95 | 1.0 / 0.95 | 1.0 / 0.95 (1.2 garbled 13% of this model's outputs in Task 2) |
| Max response | 8,192 | 2,048 | **3,072** (the model's context is 4,096) |
| Reward | +1 correct / 0 wrong | same, and −1 on every token of a truncated response | same. The truncation penalty is the released code's (`--mask_repeated_samples`), applied identically to all three methods; it hits about 1% of responses. |
| Advantage | (GA + LA)/√n | same code | same code |
| Model | Qwen2.5-14B SFT | Qwen2.5-Math-1.5B-Instruct | same (§4) |
| Training data | MATH train + NuminaMath subset | 1,432 filtered `train_30k` problems | ~5,600 filtered `train_30k` problems, eval problems removed |
| Evaluation | 6 benchmarks: MATH500, Omni-MATH-500, AIME2024, AMC, OlympiadBench, LiveCodeBench; greedy | MATH500, AMC, Omni (1,082) | **4 of the 6: + OlympiadBench (1,757)**, greedy and mean@8. Not used: AIME2024 (30 problems, not in the repo) and LiveCodeBench (code; our model and data are math-only). |
| Seeds | not stated | 1 | 1 (42); `SEED=` adds repetitions |

**GRPO is our own baseline, not part of the paper.** It uses a GRPO-style advantage, (r − mean) / std over each prompt's 16 responses, inside the same training code as ChainRL. It is not DeepSeekMath's GRPO setup, so the report should call it a "GRPO-style advantage" baseline:

| | DeepSeekMath GRPO (§4.1) | Ours |
|---|---|---|
| Group size | 64 | 16 (the same chains as ChainRL) |
| Model | DeepSeekMath-Instruct 7B | Qwen2.5-Math-1.5B-Instruct |
| Reward | reward model | 0/1 answer check |
| KL | separate loss term, β = 0.04 (unbiased "k3" estimator) | per-token reward penalty, β = 1e-4, shared with TreeRL and ChainRL |
| Truncated responses | — | −1 on every token, applied after group normalization (the shared penalty above) |
| Learning rate | 1e-6 | 1.5e-6 |

## 4. Model: keep Qwen2.5-Math-1.5B-Instruct

**Why keep it:**
- It is the model of Task 2 and the mid-submission, so those analyses stay valid in the final report.
- The whole pipeline is tested on it: memory, prompt format, grading, weight sync.
- 1.5B is the only size where paper-sized batches for five runs fit in ₹6,000. A 7B model needs about 5× the compute per response, about ₹3,500 for one TreeRL run.
- The proposal's own measures (where extra rollouts go, credit localization, PassRate at a fixed budget) are measured on trees. They don't need large accuracy gains.

**Its weakness:** it has already been through RL, so accuracy gains will be small (~1–2 points) and method differences smaller still. Two things make those differences detectable:
- OlympiadBench is added (more problems, more headroom);
- every comparison uses paired CIs on the same problems.

**The one alternative worth considering:** Qwen2.5-Math-1.5B, the base model. RL from a base model gives large, fast learning curves, which is the usual small-scale testbed for comparing RL methods. The costs:
- early gains are mostly the model learning the answer format, which every method learns equally;
- prompts and truncation behaviour would have to be re-checked on paid time;
- Task 2 would no longer match.

Only switch if a free pilot on pkgpu2 favours it. To run the pilot, score both models on the training data with `solve_rate.py` and compare accuracy, the share of problems solved 1–7 of 8, and truncation rates.

## 5. What each run logs

| File | Contents |
|---|---|
| `<ckpt>/<run>/train_log.jsonl` | One line per step. **Before:** loss, grad norm, mean leaf accuracy (`pass_at_1`), share of prompts with a correct leaf (`pass_rate`), mean nonzero per-token reward (`reward`), sampling entropy (`response_entropy`, mean −log p of sampled tokens), response length, finished share (`response_overlong_ratio`), generation/rollout time. **New:** `lr`, `train_time`, `adv_mean` and `adv_abs` (token-mean advantage the loss uses, and its magnitude), `generated_responses` and `gen_tokens` (shared prefixes counted once: the inference cost), `trained_responses` and `trained_tokens`, `truncated`, `prompts_all_correct` and `prompts_all_wrong` (these give no gradient), `resumed_from` (the step this attempt resumed from; 0 = fresh). `kl` is now non-zero. |
| `<ckpt>/<run>/trees.jsonl` (new) | One line per prompt per step, about 6 KB, about 20 MB per 200-step run:<br>• `q`: sha1 of the problem text (first 12 hex digits; to join with the data file) and the gold answer;<br>• `nodes`: every generated response, with parent, fork position (`split`, `prefix`), surprisal of the forked token (`fork_nll`), length, finish reason, 0/1 reward (`r`), and `phase`/`score` slots for your method;<br>• `paths`: every trained response's reward and its segments as `[tokens, advantage]`. |
| `logs/samplings/<run>.jsonl` (relative to the run's working directory, probably the repo root) | Prompt, response text and gold answer of every trained response, in the order of `trees.jsonl`'s paths. No step number: resumed steps repeat. |
| `<ckpt>/<run>.tryN.log`, `<ckpt>/<run>.gpu_mem.log` | Full stdout; our GPU memory each minute. |
| `<ckpt>/<run>/_actor_global_stepN/` | Weights every `SAVE_STEPS` (no optimizer state, so a resume restarts Adam; such steps carry `resumed_from` > 0, and `runs.csv` lists them). |
| `results/rl_paper_eval.csv`, `results/rl_paper_per_problem*/` | Per model and benchmark: accuracy and pass@8. Per problem: correct count, mean response length, truncations (the last two are new). |

**Which file each analysis uses:**
- Accuracy vs base and method vs method: the per-problem files → `paired.csv`.
- Same-budget comparison: `gen_tokens`.
- Where forks land (position, surprisal, Phase I vs II): `trees.jsonl` nodes.
- How credit varies along a response: `paths[].segs`.
- Data efficiency: `prompts_all_*`.
- Stability: `response_entropy`, `adv_abs`, `grad_norm`, `lr`.

**Still not logged (on purpose):** token ids and log-probs of every tree, about 50 MB per step.

**For your method:** set `node.phase` (`"I"`/`"II"`) and `node.select_score` (e.g. `{"D": novelty, "C": credit, "p": allocation probability}`) where it places a fork. They then appear in `trees.jsonl` with no other change. Scores of candidates that were *not* chosen are not captured; log them separately if the allocation-quality analysis needs them. The counterfactual oracle of the proposal is inference-only (`mid_submission_likhit/tree_alloc/oracle.py`), so it runs on pkgpu2.

## 6. GPU plan for the ₹6,000 cloud budget

**What we actually use (2026-10-28):** the A100-80GB was not available, so the runs use an **RTX PRO 6000 (Blackwell, 96 GB) at ₹208.66/hr** with storage, chosen over the H100 (₹331.78/hr).
- **Measured** at the paper batch (16 prompts × 30 leaves, GPU Adam, KL 1e-4): **1.68 min per TreeRL step**, peak 78 of 98 GB.
- **TreeRL, ChainRL and GRPO at 200 steps:** ~15 h with evaluation, ~₹3,650 including setup and the data build. That leaves ~₹2,350 (~11 h) for your method and, at most, a half-size (`ROLLOUT=8`) ablation. `STEPS=300` does not fit.
- **Setup issues met and fixed on this GPU:**
  - its CUDA 12.6 compiler can't build DeepSpeed's GPU Adam for Blackwell, so `setup_cloud.sh` installs CUDA 12.9's;
  - leftover Ray workers of finished runs kept GPU memory; the scripts now kill every process of a run;
  - the KL reference model kept 33 GB of cached memory and starved the optimizer step; it now releases it after each forward pass.

The analysis below was written before the GPU choice.

**Use the A100-80GB for all RL training.** Use pkgpu2 (free) for everything else: data filtering, smoke tests, developing and debugging your method's RL code, the inference-only ablations and the oracle.

| GPU | ₹/hr | For this project |
|---|---|---|
| **A100-80GB** | 140.94 | **Recommended.** The only option where Adam moves onto the GPU (`ADAM_OFFLOAD=`, ~28 GB). On every smaller card the optimizer stays on the CPU and the full gradient is copied there after *every* response, about half of today's training time. 80 GB also lets vLLM keep 384 forked sequences in memory without preempting. Estimated ~2.5–3× faster than our L40S runs, at a similar cost per step to an A30. |
| A30 | 38.88 | Fits only with CPU Adam and vLLM sleeping. Its ~5 GB KV cache would make vLLM preempt during the 384-sequence fork round. ~2.5–3× slower wall clock, and it depends on the instance's CPU and RAM (CPU Adam needs ≥ 48 GB RAM). Use only for inference jobs if pkgpu2 is busy. |
| A100 40GB | 84.24 | Too small for Adam on the GPU, so CPU-bound like the A30 at twice the price. |
| L4 | 41.31 | Low memory bandwidth: slow generation. |
| RTX PRO 6000 | 207.36 | ~50% more per hour than the A100-80GB. Newer architecture: needed a CUDA 12.9 compiler for DeepSpeed's Adam (see above). **Used for the runs; measured 1.68 min per TreeRL step.** |
| H100 | 330.48 | ~1.5–2× an A100 on this workload at 2.35× its price: ~18 hours for ₹6,000, too few for all methods. |

**Settings on the A100-80GB:**

```
ADAM_OFFLOAD= KEEP_VLLM=1 VLLM_MEM=0.3 NEED_RL=70000 EVAL_MEM=0.3
```

**Plan and estimated cost.** The time estimates are guesses until step 0 measures the real speed: TreeRL ~2.15 min/step, ChainRL/GRPO ~1.2 min/step, your method ~2.45 min/step (all including ~7% for the KL reference model), ~15 min to evaluate a checkpoint.

| # | Job (200 steps; evaluated at 50/100/150/200) | Hours | ₹ |
|---|---|---|---|
| 0 | Environment, smoke test, 10-step timing of TreeRL | 2 | 282 |
| 1 | TreeRL | 8.2 | 1,156 |
| 2 | ChainRL | 5.0 | 705 |
| 3 | Your method (Phase I + II, same 30 rollouts per prompt as TreeRL) | 9.2 | 1,297 |
| 4 | GRPO | 5.0 | 705 |
| 5 | One ablation of your method (e.g. the k rollouts placed at random, same n and k) | 9.2 | 1,297 |
| | **Total** | **38.6** | **5,442** |
| | Reserve: overruns, failed runs, storage | ~4 | ~558 |

**Decisions after step 0** (compare the measured TreeRL minutes per step with the 2.15 estimate):
- **≤ 1.7 min:** run jobs 1–3 with `STEPS=300`, matching the paper's curves. That costs about the whole reserve.
- **≥ 2.6 min:** run job 5 with `ROLLOUT=8`, or drop it.
- **Order:** jobs 1–3 are the core result; job 5 runs only if money is left.

**Rules that save money:**
- Develop and debug on pkgpu2 only. A crash on paid time costs ₹140 per hour of waiting.
- Pause the instance as soon as the pipeline ends. It finishes unattended, often at night. JarvisLabs documents a way to pause an instance from inside it; check their docs and append that to the pipeline command.
- Check JarvisLabs' storage price for paused instances. Delete checkpoints after evaluating them, or keep only the ones at steps 100 and 200.
- Your method: each extra *sequential* generation round costs about the time of the longest continuation (~20–40 s). Place Phase II's k rollouts in 1–2 batched rounds, not k sequential ones; ten sequential rounds would roughly double a step.
- Check that your method and TreeRL generate about the same number of tokens per prompt (`gen_tokens`). That is the "same budget" in the proposal.

## 7. How to run

**On pkgpu2 (free).** These steps run there because the cloud budget is for training only.

```bash
# 0. laptop, repo root: sync the code (as in RUN_LOG.md)
rsync -a --exclude .git --exclude .DS_Store --exclude 'mid_submission/results' --exclude '*.pdf' \
    --exclude __pycache__ --exclude 'mid_submission/data' ./ likhit@10.4.25.54:likhit/tree_based_rl/TreeRL_rl/
ssh likhit@10.4.25.54
cd ~/likhit/tree_based_rl/TreeRL_rl && source ~/likhit/.venv/bin/activate

# 1. training data: 12,000 train_30k problems, keep those solved 1-7 of 8 times, minus the eval problems
#    (~4 h on one GPU; expect ~5,600 problems, enough for 300 steps x 16 prompts)
cd mid_submission
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 nohup python solve_rate.py --data ../datasets/train/train_30k.jsonl \
    --sample 12000 --n 8 --max_tokens 3072 --gpu_mem 0.45 \
    --exclude ../datasets/eval/MATH500.jsonl ../datasets/eval/aimo-validation-amc.jsonl data/omni_math_500_seed0.jsonl \
              ../datasets/eval/olympiad_bench.jsonl \
    --keep_mixed data/train_30k_mixed_12k.jsonl --out data/train_30k_solve_rate_n8_12k.jsonl > /tmp/likhit_filter12k.log 2>&1 &
cd ..

# 2. smoke test (~15 min; GPU: one with ~35 GB free). Every check prints PASS/WARN/FAIL; it ends with
#    SMOKE TEST PASSED or FAILED. It runs TreeRL (3 steps, 2 prompts x 30 leaves, KL 1e-4: the first real run of
#    the all-leaves path, the tree log and the KL reference model), ChainRL and GRPO (1 step each) and a tiny
#    evaluation of all four benchmarks
GPU=1 KEEP_VLLM=1 VLLM_MEM=0.25 scripts/smoke_test.sh

# 3. your method: implement Phase I/II where EPTree picks forks (entropy_chain_local_manager.py), then
#    smoke-test it the same way. Then the inference-only ablations and the oracle (tree_alloc).

# 4. bring the training data to the laptop, for the copy to the cloud
exit
scp likhit@10.4.25.54:likhit/tree_based_rl/TreeRL_rl/mid_submission/data/train_30k_mixed_12k.jsonl mid_submission/data/
```

**On the A100-80GB (JarvisLabs).**

*How the code gets there: rsync over SSH.* It is the same command as for pkgpu2, and only changed files move on later syncs. The first sync is ~15 MB: results, PDFs, old work and git history stay behind.
- **GitHub** would need the work-in-progress committed to Likhit's shared repo, plus a token on the instance. The training data wouldn't travel anyway: `.gitignore` excludes `data/`.
- **JupyterLab upload** works for a one-off, but it is manual on every code change.

Commit the code before the paid runs anyway, so results can be tied to a commit.

1. **Before creating the instance:**
   - Add the laptop's public SSH key (the one pkgpu2 accepts, e.g. `~/.ssh/id_ed25519.pub`) in JarvisLabs' SSH key settings.
   - Choose disk space for the checkpoints: ~25 GB per 200-step run at `SAVE_STEPS=25` (8 × 3 GB), so ≥ 150 GB for the five jobs. With 100 GB, use `SAVE_STEPS=50`.
2. **Create** an A100-80GB instance from the PyTorch template. Note the SSH host and port it shows.
3. **Copy the code** (from the laptop, repo root; `mid_submission/data` must hold `train_30k_mixed_12k.jsonl`):

   ```bash
   rsync -avz -e "ssh -p <port>" --exclude .git --exclude .DS_Store --exclude '*.pdf' --exclude '*.zip' \
       --exclude __pycache__ --exclude 'mid_submission/results' --exclude 'mid_submission_*' --exclude ckpt \
       --exclude logs ./ root@<host>:/home/TreeRL/
   ```

4. **Set up and smoke-test** (on the instance; `tmux` keeps them alive if SSH drops):

   ```bash
   cd /home/TreeRL && bash scripts/setup_cloud.sh     # ~10-15 min, once; ends with SETUP DONE
   source /home/treerl_env.sh                          # in every new shell on the instance
   BENCH=1 scripts/smoke_test.sh                       # ~40 min: the smoke test, then 10 TreeRL steps at the
                                                       # paper batch, with minutes per step vs the plan (section 6)
   ```

   `setup_cloud.sh`:
   - installs pkgpu2's package versions (vLLM 0.30.0 and torch 2.13.0 for CUDA 12.9) into `/home/venv`;
   - downloads the model;
   - compiles DeepSpeed's GPU and CPU Adam;
   - writes `/home/treerl_env.sh` with this GPU's settings (on an 80 GB card: `ADAM_OFFLOAD= KEEP_VLLM=1 VLLM_MEM=0.3 NEED_RL=70000 EVAL_MEM=0.3`). If GPU Adam doesn't compile, it switches the env file to CPU Adam (slower, but correct).

   Everything lives under `/home`, which JarvisLabs keeps while the instance is paused. The exact package versions go to `/home/ckpt/env_freeze.txt`.
5. **Launch** (after SMOKE TEST PASSED and the step-0 decisions in section 6):

   ```bash
   source /home/treerl_env.sh && cd /home/TreeRL
   nohup scripts/run_paper.sh > /home/paper.log 2>&1 &   # jobs 1, 2, 4 (job 3 once your method has a case)
   tail -1 /home/ckpt/qwen1.5b-paper-*/train_log.jsonl    # progress
   ```

6. **Bring results back** (laptop) before stopping the instance:

   ```bash
   rsync -avz -e "ssh -p <port>" root@<host>:/home/TreeRL/mid_submission/results/ mid_submission/results/
   rsync -avz -e "ssh -p <port>" --include '*/' --include '*.jsonl' --include 'run_config' --include '*.gpu_mem.log' \
       --include 'env_freeze.txt' --exclude '*' root@<host>:/home/ckpt/ mid_submission/results/rl_paper_ckpt_logs/
   ```

   The second command copies every run's logs and settings, but not the 3 GB checkpoints or the large stdout logs.

The last lines of `paper.log` are `PAPER_RUNS_DONE`, or `PAPER_RUNS_INCOMPLETE` followed by the stages that didn't finish (exit status 1); fix the cause and rerun the same command to continue. A line `note: <run> resumed from step(s) ...` names a run that continued with a fresh optimizer state; report it. A second seed is `SEED=1 nohup scripts/run_paper.sh > paper_seed1.log 2>&1 &`, which runs next to the seed-42 runs, not over them.

Knobs of `run_paper.sh`: `STEPS` (200), `ROLLOUT` (16), `LR` (1.5e-6), `KL` (1e-4), `SEED` (42), `SAVE_STEPS` (25), `EVAL_EVERY` (50), `ONLY`, `GPU`, `CKPT`, `DATA`, `VENV`, `VLLM_MEM`, `KEEP_VLLM`, `EVAL_MEM`, `NEED_RL`, `NEED_EVAL`.

**Build the tables (laptop),** after step 6:

```bash
cd mid_submission
R=results/rl_paper L=results/rl_paper_ckpt_logs; mkdir -p $R/train_logs $R/gpu_mem $R/trees
cp results/rl_paper_eval.csv $R/eval.csv && cp -R results/rl_paper_per_problem $R/per_problem
for d in $L/qwen1.5b-paper-*/; do run=$(basename $d)
  cp $d/train_log.jsonl $R/train_logs/$run.jsonl; cp $d/trees.jsonl $R/trees/$run.jsonl
  cp $L/$run.gpu_mem.log $R/gpu_mem/$run.log 2> /dev/null
done
python -m tree_alloc.rl_report $R     # eval_table.csv, train_curves.csv, runs.csv, paired.csv in results/rl_paper
```

Run names: `qwen1.5b-paper-treerl-6-2-1-2-seed42`, `qwen1.5b-paper-chainrl-16-seed42`, `qwen1.5b-paper-grpo-16-seed42`.

## 8. What to look for

- **TreeRL `train_log.jsonl`:**
  - `adv_mean` and `reward` should average about 0, not +0.04 to +0.10;
  - `trained_responses` = `generated_responses` = 480 per step.
- **Sampling entropy** (`response_entropy`):
  - if it climbs steadily for any method, as ChainRL's did at lr 5e-6, rerun that method with `LR=1e-6`;
  - if it collapses towards 0, the policy is becoming deterministic.
- **`kl`:** small and positive, growing slowly. A sudden jump means the policy is moving away from the start fast.
- **Budget match:** `gen_tokens` per step of TreeRL, ChainRL and your method should be within ~20% of each other.
- **The comparison that matters:** each method minus ChainRL at steps 100–200, from `paired.csv` (pooled over 1,757 problems). Expect gains over the base model of ~1–2 points at most (cause 5); the paired CIs, not single scores, decide.
