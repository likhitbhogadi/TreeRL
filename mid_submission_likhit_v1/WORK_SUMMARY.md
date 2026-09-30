# Work summary: Tier 1, Tasks 1–2

What was done, what it found, and every file added or changed. Full results are in [tier1_implementation.md](tier1_implementation.md), and a step-by-step code explanation is in [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md). This page is the map.

**Status:**
- Task 1 (harness) and Task 2 (EPTree replication) are done.
- **All trees are built by TreeRL's own released code**, run standalone after small edits.
- Tasks 3–6 (our Phase I / Phase II methods) are out of scope for now.

---

## 1. What was done, in order

| # | Step | Outcome |
|---|---|---|
| 1 | Reviewed the proposal and the Tier-1 task list against the TreeRL paper and code | Scope decisions |
| 2 | Set up pkgpu2: uv env, vLLM for CUDA 12.9, model download | `torch 2.13.0+cu129`, `vllm 0.30.0+cu129` |
| 3 | First built our own tree sampler and ran Task 2 with it (OlympiadBench, then Omni-MATH) | Found: filtered OlympiadBench saturates PassRate, so switched to unfiltered Omni-MATH; **temperature 1.2 → 1.0** (1.2 made 13% of outputs degenerate) |
| 4 | Made TreeRL's own code run standalone (small edits to 3 files) and cross-checked it against ours | Identical fork selection on 300/300 chains; statistically the same outcomes |
| 5 | Switched to TreeRL's code for all tree building; deleted our sampler | `run_treerl.py` drives their manager; our code keeps only I/O, grading and analysis |
| 6 | Re-ran Task 1 checks and the full Task 2 sweep with TreeRL's code (500 Omni-MATH problems, 11 configs) | Results below |
| 7 | Clean-up | Removed Tasks 3–6 code and plan, superseded results, temporary files, and optional extras (smoke-test / check scripts, add-on analyses, unit tests) |

Two problems hit along the way, both fixed:
- **Grading hang.** TreeRL's grader runs in threads, where a symbolic checker can hang forever. It is now a string check; reported numbers are re-graded in the main thread.
- **Shared-GPU collisions.** Workers now wait for free GPU memory instead of failing.

## 2. Key results (TreeRL's code, 500 unfiltered Omni-MATH problems)

Qwen2.5-Math-1.5B-Instruct, T = 1.0, top-p 0.95. i.i.d. PassRate: 44.4% at 16 chains, 53.2% at 64 (paper, 14B model: 52.4% and 67.4%).

| Paper claim | Our result |
|---|---|
| Trees beat i.i.d. sampling at the same token cost (Fig. 5) | **Replicated.** Every tree config sits above the i.i.d. curve, by up to about +3 points at matched tokens. Random forking gains as much as EPTree, so the gain comes from shared prefixes |
| Entropy forking beats random forking (Table 2) | **No detectable difference:** +0.4, +0.8, 0.0, −0.8, and all CIs include 0. The paper's +2.1 / +1.0 are inside our CIs. EPTree does give slightly more distinct answers |
| Fork positions roughly uniform (Fig. 8) | **Replicated** (mean relative position 0.50) |
| Forking tokens (Fig. 7) | Similar: ` the`, ` \(`, ` and`, ` we`. No "Wait"/"But", which our model doesn't write |


## 3. Files added (all under `mid_submission/`)

| File | Purpose |
|---|---|
| `run_treerl.py` | Runs TreeRL's tree builder per problem; converts, grades, saves; resumable; sharded |
| `task2.sh` | Runs all 11 Task 2 configs in parallel shards (GPU-memory aware), merges, runs the reports |
| `tree_alloc/gen.py` | Loads the model in vLLM (raw logprobs, chat template) |
| `tree_alloc/data.py`, `verify.py` | Problem loaders; `\boxed{}` grading |
| `tree_alloc/tree.py` | Tree log format (JSON) |
| `tree_alloc/run.py` | JSONL helpers, summary table |
| `tree_alloc/metrics.py` | Per-tree metrics (summary table); paired bootstrap |
| `tree_alloc/task2_report.py` | Figs 4/5/7/8, Table 2 |
| `tier1_implementation.md`, `CODE_WALKTHROUGH.md`, `WORK_SUMMARY.md` | Write-up, code walkthrough, this page |
| `results/task2/` | Task 2 results: tables and figures |
| `requirements.txt`, `.gitignore` | GPU-machine dependencies; ignores `logs/` |

`tier1_inference_plan.md` (your original plan) was trimmed to Tasks 1–2.

## 4. Files changed in the TreeRL repo

Three files; the algorithm is unchanged. `git diff openrlhf` shows everything.

| File | Change |
|---|---|
| `openrlhf/trainer/ppo_utils/evaluation.py` | (1) With no Ray actor, call a local `vllm.LLM` via `TokensPrompt` (vLLM 0.30 API). (2) With no judge URLs, `check_result` does a normalized string match of the last `\boxed{}`, which can't hang in threads |
| `openrlhf/trainer/ppo_utils/parallel_mcts.py` | Empty judge/RM URL lists when the authors' config files (`/workspace/lurui/...`) don't exist, instead of crashing on import |
| `openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py` | New `random_fork` flag (the paper's random-forking ablation, missing from the released code) |

## 5. Where things live

| | Local (`~/Desktop/anlp/TreeRL/mid_submission`) | pkgpu2 (`likhit@10.4.25.54`) |
|---|---|---|
| Code and docs | Yes | Yes, in `~/likhit/tree_based_rl/TreeRL/mid_submission_likhit/` (identical) |
| Raw logs (JSONL trees, 1.4 GB) | No | `mid_submission_likhit/logs/` (`task2/`, smoke and temperature tests) |
| Omni-MATH sample | No | `mid_submission_likhit/data/omni_math_500_seed0.jsonl` |
| Results | `results/task2/` | `mid_submission_likhit/results/task2/` |

Environment (`~/likhit/.venv`, uv):
- `torch 2.13.0+cu129` (the original cu130 build can't use the GPU with driver 575), `vllm 0.30.0+cu129`, `math-verify`, `matplotlib`;
- `levenshtein`, `multiprocess`, `ipython` and `ray` for TreeRL's code.

SSH: your laptop's key is installed on pkgpu2, so `ssh likhit@10.4.25.54` works without a password. pkgpu2 is reachable only from the campus network or VPN.

## 6. How to reproduce (on pkgpu2)

```bash
cd ~/likhit/tree_based_rl/TreeRL/mid_submission_likhit && source ~/likhit/.venv/bin/activate
nohup ./task2.sh > /tmp/task2.log 2>&1 &                    # Task 2 (resumable; GPUS=0 SLOTS=3 to limit)
python -m tree_alloc.run logs/task2/*.jsonl                 # summary table
```

## 7. Open items

- **Optional:** sample 128 i.i.d. chains so the (8,4,2,2) tree has an exact token-matched baseline; add seeds to settle entropy vs. random.
- **Nothing is committed yet.** Locally and on pkgpu2, the changes above are uncommitted in git.
