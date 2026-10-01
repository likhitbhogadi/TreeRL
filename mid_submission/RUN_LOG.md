# Run log: commands and scripts (Task 1 CSVs, RL port, baselines)

Every command run for this part of the project, in order, with what it produced. Code changes are explained in [RL_CHANGES.md](RL_CHANGES.md) and [GRPO_CHANGES.md](GRPO_CHANGES.md); results are in [results/rl/ANALYSIS.md](results/rl/ANALYSIS.md).

**Machines and folders**
- **Laptop:** the git repo, `~/Desktop/anlp/TreeRL` (branch `sanyam`, pushed to `github.com/likhitbhogadi/TreeRL`).
- **pkgpu2** (`ssh likhit@10.4.25.54`): 2× L40S (46 GB each), shared with other users. Python env `~/likhit/.venv` (uv). Three folders under `~/likhit/tree_based_rl/`:
  - `TreeRL/`: Likhit's checkout. Holds the raw Task 2 logs (`mid_submission_likhit_v1/logs`, 1.4 GB).
  - `TreeRL_rl/`: **the RL working copy.** It is synced from the laptop with the `rsync` command below and is not a git repo. All RL runs, checkpoints (`ckpt/`) and eval results live here.
  - `v2/`: a clean clone of the GitHub repo. Unused.

The code is synced to the server before every run:

```bash
# laptop, repo root; results/ and data/ are excluded so the server's own outputs are never overwritten
rsync -a --exclude .git --exclude .DS_Store --exclude 'mid_submission/results' --exclude '*.pdf' \
    --exclude __pycache__ --exclude 'mid_submission/data' ./ likhit@10.4.25.54:likhit/tree_based_rl/TreeRL_rl/
```

All server commands below run in `~/likhit/tree_based_rl/TreeRL_rl` after `source ~/likhit/.venv/bin/activate`, unless a step says otherwise.

## 1. Task 1: tree logs → CSV

Script: `mid_submission/tree_alloc/to_csv.py` (new).

```bash
# server, in TreeRL/mid_submission_likhit_v1 (where the logs are)
python -m tree_alloc.to_csv --logs logs --out results/csv --results results/task2/results.md
# laptop
scp 'likhit@10.4.25.54:likhit/tree_based_rl/TreeRL/mid_submission_likhit_v1/results/csv/*' mid_submission/results/csv/
```

Output: `results/csv/trees.csv` (5,600 trees), `nodes.csv` (one row per leaf, 33 MB), and the 5 `results.md` tables as CSVs. Checked: the CSV reproduces `results.md` (e.g. 48.8% PassRate for (6,2,1,2)).

## 2. Environment for RL training

```bash
~/.local/bin/uv pip install --dry-run deepspeed datasets accelerate   # check: only additions, torch/vLLM untouched
~/.local/bin/uv pip install deepspeed datasets accelerate               # deepspeed 0.19.7, datasets 5.0.1, accelerate 1.15.0
# DeepSpeed CPU Adam builds and runs (system nvcc 12.0 vs torch cu129 is accepted):
python -c "import torch; from deepspeed.ops.adam import DeepSpeedCPUAdam; p = torch.nn.Parameter(torch.zeros(10)); \
o = DeepSpeedCPUAdam([p], lr=1e-3); p.grad = torch.ones(10); o.step(); print(p[:3])"
hf download Qwen/Qwen2.5-0.5B-Instruct                                  # small model for smoke tests
```

The installed stack: torch 2.13.0+cu129, vLLM 0.30.0, transformers 5.17.0, Ray 2.58.0.

## 3. RL port: checks and smoke tests

Scripts:
- `scripts/treerl-qwen1.5b-1gpu.sh` (new): the launcher. Its knobs are environment variables: `GPU STEPS ROLLOUT NUM_TRACE TREE ADV LR MAX_LEN DATA VLLM_MEM INFER_BS MODEL TAG SAVE_DIR SAVE_STEPS RESUME`.
- `scripts/check_vllm_weight_sync.py` (new).
- The code changes are listed in RL_CHANGES.md.

```bash
# imports + Qwen prompt format (must equal Task 2's prompt)
PYTHONPATH=. python -c "import train_reinforce_ray; from openrlhf.trainer.ppo_utils.experience_maker import _tokenize_fn_llama"

# actor -> vLLM weight sync over CUDA IPC (~2 min, ~7 GB)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/check_vllm_weight_sync.py
#   -> max |logprob gap| vs HF: same weights 0.040, after perturbing HF only 0.491, after sync 0.007 / weight sync OK

# end-to-end smoke runs on 0.5B (they had to fit the ~11 GB other users left free)
GPU=0 STEPS=2 ROLLOUT=4 VLLM_MEM=0.12 INFER_BS=1 SAVE_STEPS=1000 MODEL=Qwen/Qwen2.5-0.5B-Instruct TAG=smoke-0.5b \
    SAVE_DIR=/tmp/likhit_rl_smoke nohup scripts/treerl-qwen1.5b-1gpu.sh > /tmp/likhit_rl_smoke.log 2>&1 &
# same with STEPS=1 SAVE_STEPS=1: checkpoint saved, reloads with no missing keys
# resume: copy a checkpoint into a dir whose name contains "qwen", then RESUME=1 STEPS=2 -> "past_steps: 1", step 2 trained
# ChainRL path: TREE="4 0 0 0" NUM_TRACE=4 MAX_LEN=2048 ...       GRPO path: TREE="4 0 0 0" ADV=grpo MAX_LEN=1024 ...
```

The smoke runs found and fixed:
- several bugs in the released code (R1 chat marker on Qwen prompts, string stop tokens, a judge URL of `None`, ragged rewards, tied-embedding save);
- one OOM, fixed by sleeping vLLM after each rollout;
- a slow grad-norm loop (3.7 → 1.5 s per sample).

Details are in RL_CHANGES.md.

## 4. Training data: difficulty filter

Script: `mid_submission/solve_rate.py` (new; it also does the evaluation).

```bash
# server, in TreeRL_rl/mid_submission; GPU1, ~60 min
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 nohup python solve_rate.py --data ../datasets/train/train_30k.jsonl \
    --sample 3000 --n 8 --max_tokens 2048 --gpu_mem 0.45 \
    --keep_mixed data/train_30k_mixed.jsonl --out data/train_30k_solve_rate_n8.jsonl > /tmp/likhit_filter.log 2>&1 &
# -> 2,998 problems | mean acc 0.320 | pass@8 0.577 | mixed 1,432 (solved 1-7 of 8)
scp 'likhit@10.4.25.54:likhit/tree_based_rl/TreeRL_rl/mid_submission/data/train_30k_*.jsonl' mid_submission/data/   # laptop
```

## 5. Baseline pipeline

Script: `scripts/run_baselines.sh` (new). It runs, in order:
1. base-model eval;
2. TreeRL, ChainRL, GRPO at 40 steps, each followed by greedy eval of its checkpoints;
3. 8-sample eval at steps 20 and 40;
4. long runs (150 steps, lr 5e-6) of all three, each followed by 8-sample eval.

How it behaves:
- Each stage waits for a GPU with enough free memory (`NEED_RL` 21 GB, `NEED_EVAL` 14 GB) and is skipped if already done (markers in `ckpt/.done/`).
- A failed or stalled training run (45 min without a step) is retried up to 3 times, resuming from its newest checkpoint.
- It logs our GPU memory to `ckpt/<run>.gpu_mem.log`.

The underlying commands it runs:

```bash
# training (one per run; values for the 40-step runs; long runs add STEPS=150 LR=5e-6 SAVE_STEPS=50)
env ROLLOUT=8 NUM_TRACE=8 MAX_LEN=2048 STEPS=40 SAVE_STEPS=10 VLLM_MEM=0.3 GPU=<free gpu> TREE="6 2 1 2" \
    TAG=qwen1.5b-treerl-6-2-1-2 SAVE_DIR=ckpt/qwen1.5b-treerl-6-2-1-2 DATA=mid_submission/data/train_30k_mixed.jsonl \
    scripts/treerl-qwen1.5b-1gpu.sh            # ChainRL: TREE="8 0 0 0"; GRPO: TREE="8 0 0 0" ADV=grpo
# evaluation (from mid_submission/); greedy, or add --n 8 --temperature 1.0 for mean@8
CUDA_VISIBLE_DEVICES=<free gpu> python solve_rate.py --model <checkpoint dir or HF id> --temperature 0 --gpu_mem 0.28 \
    --summary ckpt/.done/eval_<name>.csv \
    --data ../datasets/eval/MATH500.jsonl ../datasets/eval/aimo-validation-amc.jsonl data/omni_math_500_seed0.jsonl
```

The full training command, as run (from `ckpt/qwen1.5b-grpo-8.try1.log`):

```
python train_reinforce_ray.py --actor_num_nodes 1 --actor_num_gpus_per_node 1 --ref_num_nodes 1 --ref_num_gpus_per_node 1
  --reward_num_nodes 0 --vllm_num_engines 1 --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.3
  --enable_prefix_caching --pretrain Qwen/Qwen2.5-Math-1.5B-Instruct --reward_pretrain Qwen/Qwen2.5-Math-1.5B-Instruct
  --save_path <ckpt dir> --ckpt_path <ckpt dir> --save_steps 10 --micro_train_batch_size 1 --train_batch_size 64
  --micro_rollout_batch_size 8 --rollout_batch_size 8 --inference_batch_size 2 --num_episodes 1 --max_samples 320
  --prompt_max_len 1024 --generate_max_len 2048 --zero_stage 2 --adam_offload --bf16 --gradient_checkpointing
  --actor_learning_rate 1.5e-6 --lr_scheduler_type cosine --min_actor_learning_rate_lr 1 --l2 0.1 --init_kl_coef 0
  --prompt_data <abs path>/train_30k_mixed.jsonl,1 --input_key text --label_key label --source_key data_type
  --top_p 0.95 --temperature 1.0 --num_trace_per_sample 8 --task_type qwen-math-reinforce
  --remote_rm_url <abs path>/scripts/remote_reward_url.json --advantage_estimator grpo --use_mcts --use_entropy_tree
  --m 8 --n 0 --l 0 --t 0 --process_supervision --use_state_value_reward --use_pure_binary --use_weighted_value
  --weighted_value_style sqrt --mask_repeated_samples --correct_bonus_ratio 1 --correct_bonus_threshold 0 --perf
  --wandb_run_name qwen1.5b-grpo-8
```

Launches and restarts (server time, 2026-09-30):

```bash
nohup scripts/run_baselines.sh > /tmp/likhit_baselines.log 2>&1 &         # 15:22, first launch
# 19:15: threshold 26 -> 21 GB, eval right after each run. Restarted while it was only waiting:
for p in $(pgrep -u likhit -f "run_baseline[s].sh"); do kill $p; done
nohup scripts/run_baselines.sh >> /tmp/likhit_baselines.log 2>&1 &
# 19:49: sampled eval + long runs added; queued behind the running instance (PID 1900702) instead of in parallel:
nohup sh -c "while kill -0 1900702 2>/dev/null; do sleep 60; done; exec scripts/run_baselines.sh" \
    >> /tmp/likhit_baselines.log 2>&1 &
# 20:0x: the GRPO stage and the long ChainRL/GRPO runs were added to the script file before the queued instance started.
```

`rsync` replaces files by renaming them, so syncing never disturbs a running instance. A running `bash` keeps reading the old file.

Monitoring:

```bash
ssh likhit@10.4.25.54 'tail -5 /tmp/likhit_baselines.log; cat ~/likhit/tree_based_rl/TreeRL_rl/mid_submission/results/rl_eval.csv'
ssh likhit@10.4.25.54 'tail -1 ~/likhit/tree_based_rl/TreeRL_rl/ckpt/*/train_log.jsonl'
```

### 2026-10-01: reruns A + B

```bash
# keep-memory smoke test (0.5B, 1 step): trained, 0 "fall asleep" lines
GPU=0 STEPS=1 ROLLOUT=2 NUM_TRACE=2 TREE="2 0 0 0" KEEP_VLLM=1 MAX_LEN=512 VLLM_MEM=0.12 INFER_BS=1 SAVE_STEPS=1000 \
    MODEL=Qwen/Qwen2.5-0.5B-Instruct TAG=keep-smoke SAVE_DIR=/tmp/likhit_qwen_keep_smoke \
    nohup scripts/treerl-qwen1.5b-1gpu.sh > /tmp/likhit_keep_smoke.log 2>&1 &
# updated pipeline queued behind the running instance (PID 1951404, long GRPO attempt 3)
nohup sh -c "while kill -0 1951404 2>/dev/null; do sleep 60; done; exec scripts/run_baselines.sh" \
    >> /tmp/likhit_baselines.log 2>&1 &
# afterwards (laptop): copy results + per-problem files, rebuild tables incl. paired.csv
scp -q -r likhit@10.4.25.54:likhit/tree_based_rl/TreeRL_rl/mid_submission/results/rl_per_problem mid_submission/results/rl/per_problem
cd mid_submission && python -m tree_alloc.rl_report results/rl
```

## 6. Other one-off checks

```bash
# TreeRL's advantage code on fake chains/trees (server CPU): chains get ±1.33 (RLOO x2); a fork gets per-segment credit
# GRPO advantage on fake groups: [1,0,0,1] -> ±1.0; 1 of 8 -> +2.646 / -0.378; all wrong -> 0; guard rejects --l > 0
#   (inline Python run once with CUDA_VISIBLE_DEVICES= PYTHONPATH=.; not kept as files; results in RL_CHANGES.md / GRPO_CHANGES.md)
# per-problem stats from the Task 2 CSVs (laptop): response lengths (median 682, 1.1% > 2048 tokens),
# share of all-right/all-wrong trees, string-grader vs math_verify agreement (string match misses 14.4%)
```

## 7. Collecting results (laptop)

```bash
R=likhit@10.4.25.54:likhit/tree_based_rl/TreeRL_rl
scp -q $R/mid_submission/results/rl_eval.csv mid_submission/results/rl/eval.csv
for run in qwen1.5b-{treerl-6-2-1-2,chainrl-8,grpo-8}{,-lr5e-6-150}; do
  scp -q $R/ckpt/$run/train_log.jsonl mid_submission/results/rl/train_logs/$run.jsonl
  scp -q $R/ckpt/$run.gpu_mem.log mid_submission/results/rl/gpu_mem/$run.log
done
scp -q likhit@10.4.25.54:/tmp/likhit_baselines.log mid_submission/results/rl/pipeline.log
cd mid_submission && python -m tree_alloc.rl_report results/rl      # -> eval_table.csv, train_curves.csv, runs.csv
```

## 8. Commits (branch `sanyam`)

| Commit | Content |
|---|---|
| `111f3c8` | Task 2 logs as CSV; TreeRL RL training on one GPU |
| `3bf7073` | Merge of Likhit's `_v1` folders (restoring 6 `mid_submission/` files that commit deleted) |
| `2407bbd` | Baselines: difficulty filter, ChainRL, checkpoint eval |
| `e7e5662` | Baseline pipeline and filtered RL training data |
| `2bec37e` | First 1.5B TreeRL run: measured VRAM and time; pipeline tweaks |
| `f4eb4b4` | Load the actor before starting colocated vLLM (the startup race) |
| `930b87c` | Pipeline: sampled eval and a longer TreeRL run |
| `e222155` | GRPO baseline |
| `7c6236a` | Queue long ChainRL and GRPO runs |
| (this commit) | Results folder, `rl_report.py`, ANALYSIS.md, this log, and a pipeline logging fix |

## Scripts written or changed

| Script | Purpose |
|---|---|
| `mid_submission/tree_alloc/to_csv.py` | Task 2 JSONL tree logs → CSV |
| `mid_submission/solve_rate.py` | Sample + grade n answers per problem: the difficulty filter and checkpoint evaluation |
| `mid_submission/tree_alloc/rl_report.py` | RL results → CSV tables |
| `mid_submission/tree_alloc/data.py` | `Problem` keeps its original JSON row |
| `scripts/treerl-qwen1.5b-1gpu.sh` | One RL run on one GPU (paper flags, knobs as env vars) |
| `scripts/run_baselines.sh` | Unattended pipeline (waits for memory, retries, resumes, evaluates) |
| `scripts/check_vllm_weight_sync.py` | Self-check for the CUDA-IPC weight sync |
| `train_reinforce_ray.py`, `openrlhf/**` | The single-GPU / vLLM 0.30 port and GRPO, see RL_CHANGES.md and GRPO_CHANGES.md |
