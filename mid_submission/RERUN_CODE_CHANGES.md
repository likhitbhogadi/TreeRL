# Code changes for the paper-setting reruns

Every code change made on 2026-10-28 for rerunning TreeRL, ChainRL and GRPO with the TreeRL paper's settings ([RERUN_PAPER.md](RERUN_PAPER.md)). They come after commit `08de000` (the mid-submission merge) and are not committed yet. The earlier changes (the single-GPU port, GRPO) are in [RL_CHANGES.md](RL_CHANGES.md) and [GRPO_CHANGES.md](GRPO_CHANGES.md).

- 15 files modified (`git diff 08de000 --stat`), 6 files new.
- **The algorithm is unchanged:** tree search, advantage computation and loss (`tree_node.py`, `parallel_mcts.gather_paths`, `models/loss.py`).
- **Changes that affect results:**
  - training on all tree leaves, and 16 chains for ChainRL and GRPO;
  - KL coefficient 1e-4;
  - a 10-step learning-rate warmup.

  They are all in §1.
- **Everything else** fixes failures, adds logging, cleans the data, or runs and checks the pipeline.

## 1. Training settings (these change results)

| File | Change | Why |
|---|---|---|
| `scripts/treerl-qwen1.5b-1gpu.sh` | `NUM_TRACE` now defaults to all leaves of the tree, M(1 + N·L·T): 30 for (6,2,1,2), 16 for 16 chains. When every leaf is trained on, it passes `--use_all_terminals`, the released code's flag for this. | The paper trains TreeRL on all 30 leaves. Its Table 3 shows that 16 of 30 costs 3.2 points. Training on a subset also biased the advantages: the released code's subset always contains one correct leaf, which gave every response a mean advantage of about +0.06. That matches the +0.04 to +0.10 in our earlier logs. |
| `scripts/treerl-qwen1.5b-1gpu.sh` | KL coefficient: knob `KL`, default **1e-4** (was hard-coded 0). | 1e-4 is the paper's value; the released script used 0. A value above 0 loads a reference model (+3 GB, about +7% per step). |
| `scripts/treerl-qwen1.5b-1gpu.sh` | Knob `SEED` (default 42), passed as `--seed`. | Before, the seed was always the trainer's default of 42. It controls vLLM sampling, data order and torch seeds, and repetitions need it. |
| `openrlhf/trainer/ray/reinforce_actor.py` | The step count behind the 5% learning-rate warmup now counts optimizer steps: responses ÷ responses per step. | It divided prompt batches by micro-batches. That gave 2 "steps" for the earlier long runs, so warmup was one step at lr 0. At batch 480 it gave 0 steps, so no warmup at all. Now a 200-step run warms up over 10 steps, then holds lr 1.5e-6 (constant, as before). |

The other settings (16 prompts per step, lr 1.5e-6, 200 steps, 3,072-token responses, temperature 1.0) are passed by `scripts/run_paper.sh` (§5).

## 2. Fixes

| File | Change | Why |
|---|---|---|
| `train_reinforce_ray.py` | The KL reference model loads the starting model (`args.ref_pretrain`), also after a resume. | It loaded `args.pretrain`, which `--resume` replaces with the newest checkpoint, so the reference moved after every interruption. Matters only with KL > 0. |
| `openrlhf/trainer/ray/launcher.py` | `ReferenceModelRayActor.forward` calls `torch.cuda.empty_cache()` after each forward pass. | Found on the RTX PRO 6000: the reference model kept 33 GB of cached memory between passes, and the actor's optimizer step then ran out of memory. Memory only; results are unchanged. |
| `scripts/run_paper.sh`, `scripts/smoke_test.sh` | After each run, every process carrying the run's `TREERL_RUN` marker is killed (`kill_run`). | Ray's worker processes leave the launcher's process group and outlived their runs, holding ~75 GB of GPU memory. The next run then ran out of memory. Only our own processes of that run are killed. |

## 3. Logging

| File | Change |
|---|---|
| `openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py` | New `node_summary()`: per generated response, its parent, fork position, prefix length, length, finish reason, 0/1 reward, the surprisal of the forked token, and `phase`/`score` slots for our method. |
| `openrlhf/trainer/ppo_utils/entropy_guided_tree_search.py` | `parallel_entropy_guided_tree(..., info=None)` returns that summary in `info["nodes"]`. |
| `openrlhf/trainer/ppo_utils/experience_maker.py` | New `tree_record()`: one record per prompt with the problem hash, gold answer, all nodes, and every trained response's reward and segments as `[tokens, advantage]`, plus token and response counts. Filled for each prompt of the rollout batch. |
| `openrlhf/trainer/ppo_utils/experience_maker_reinforce.py` | Per-response `adv_mean` and `adv_abs`: the token mean of the advantage the loss uses, and its magnitude. |
| `openrlhf/trainer/reinforce_trainer.py` | Writes the records to `<run>/trees.jsonl` with the step number. Adds to `train_log.jsonl`: `lr`, `train_time`, `generated_responses`, `gen_tokens` (shared prefixes counted once: the inference cost), `trained_responses`, `trained_tokens`, `truncated`, `prompts_all_correct`, `prompts_all_wrong` and `resumed_from` (> 0 on steps trained after a resume, with a fresh optimizer). |
| `mid_submission/solve_rate.py` | Per-problem evaluation output also records the mean response length and the number of truncated responses. |

Every logging step is wrapped so that a logging error can't stop a run. The analyses each field serves are in [RERUN_PAPER.md §5](RERUN_PAPER.md#5-what-each-run-logs).

## 4. Training data

| File | Change | Why |
|---|---|---|
| `mid_submission/solve_rate.py` | New `--exclude FILE...`: drops training problems whose text (ignoring whitespace and case) appears in the given files. | `train_30k` contains 454 of the 500 Omni-MATH-500 test problems and 98 of the 675 OlympiadBench ones. The new training file (`data/train_30k_mixed_12k.jsonl`, 5,504 problems) excludes all four test sets: 226 problems dropped from the 12,000 sampled. |

## 5. Running the experiments

| File | Change |
|---|---|
| `scripts/treerl-qwen1.5b-1gpu.sh` | `VENV` (which Python environment to use) and `ADAM_OFFLOAD` (empty = Adam on the GPU, for 80–96 GB cards). `RAY_ADDRESS=local`, so each run starts its own Ray cluster. It refuses settings that would otherwise go silently wrong:<br>• `NUM_TRACE` larger than the tree's leaves;<br>• an `INFER_BS` that doesn't divide `NUM_TRACE` (the remainder would be dropped);<br>• a data file with fewer than `STEPS × ROLLOUT` rows (the run would stop early). |
| `scripts/run_paper.sh` (new) | The pipeline for the three methods at the paper's settings:<br>• waits for GPU memory, trains, resumes a failed or stalled run from its newest checkpoint;<br>• evaluates every 50th checkpoint, greedy and with 8 samples per problem, on MATH500, AMC, Omni-MATH-500 and OlympiadBench (1,757 problems).<br>Safeguards:<br>• run names end in `-seed<N>`;<br>• each run folder stores its settings in `run_config`, and a folder with other settings is never resumed, reused or evaluated;<br>• an evaluation two instances both need runs once, under a lock.<br>It ends with `PAPER_RUNS_DONE`, or with `PAPER_RUNS_INCOMPLETE`, the unfinished stages and exit status 1, and it names resumed runs. |
| `scripts/setup_cloud.sh` (new) | One-time setup of a cloud instance:<br>• pkgpu2's package versions (torch 2.13.0 and vLLM 0.30.0 for CUDA 12.9, transformers 5.17.0, DeepSpeed 0.19.7, Ray 2.58.0) and the model;<br>• on Blackwell GPUs, installs CUDA 12.9's compiler, since 12.6 can't compile DeepSpeed's GPU Adam for them;<br>• builds DeepSpeed's GPU and CPU Adam;<br>• writes `/home/treerl_env.sh` with the GPU's settings. |
| `scripts/smoke_test.sh`, `scripts/smoke_check.py` (new) | A ~15-minute smoke test with automatic PASS/WARN/FAIL checks:<br>• TreeRL (3 steps, all 30 leaves, KL 1e-4), ChainRL, GRPO and a small evaluation;<br>• after each run, nothing of ours may be left on the GPU;<br>• `BENCH=1` adds a 10-step TreeRL timing run at the paper batch. |
| `scripts/watch_and_pause.sh` (new) | Runs on the laptop: when the instance's pipeline ends, copies results and logs back, then pauses the instance with the JarvisLabs CLI (`jl pause`). |
| `scripts/run_baselines.sh` | Passes `KL=0`, so it still reproduces the earlier runs now that the launcher defaults to 1e-4. |

## 6. Reporting

| File | Change |
|---|---|
| `mid_submission/tree_alloc/rl_report.py` | Handles OlympiadBench, the new training-log fields, and which steps each run resumed from (`runs.csv`). The earlier results regenerate unchanged; the new columns stay empty. |

## 7. Documentation

- `RERUN_PAPER.md` (new):
  - why the earlier runs barely improved;
  - the settings versus the paper (a scaled replication);
  - the model choice, the logs and the GPU plan;
  - how to run everything.
- `RL_CHANGES.md` and `GRPO_CHANGES.md`: pointers to the reruns. GRPO is described as a GRPO-style advantage baseline, not DeepSeekMath's exact setup.
- `RUN_LOG.md`: the JarvisLabs session, including the three problems above and their fixes.

## 8. How the changes were checked

**Locally, without a GPU:**
- The released advantage code on simulated trees confirms the bias from training on a subset of leaves.
- The tree-log code passes its tests on simulated trees.
- Every launch command parses with the trainer's own argument parser and validation.
- The pipeline was rehearsed in 7 scenarios with stand-ins (run twice, settings clash, new seed, failed training, failed evaluation, crash and resume), plus two instances in parallel and a stale lock.
- The smoke checker passes correct outputs and fails broken ones.

**On the RTX PRO 6000:**
- All smoke tests passed.
- The 10-step timing run: 1.68 min per TreeRL step, peak 78 of 98 GB of GPU memory.
- The real TreeRL run so far: 480 responses generated and trained per step, KL active, mean advantage ≈ 0, no restarts.
