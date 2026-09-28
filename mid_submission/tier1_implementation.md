# Tier 1 Implementation: Tasks 1–2 (EPTree replication with TreeRL's code)

Companion to [tier1_inference_plan.md](tier1_inference_plan.md) and the proposal (`ANLP_final_proposal (1).pdf`). For a step-by-step explanation of the code, see [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).

**Scope: Task 1 (harness) and Task 2 (replicate TreeRL's EPTree sampling results).** Our Phase I / Phase II methods (former Tasks 3–6) are out of scope for now.

**Trees are built by TreeRL's own released code** (`EntropyGuidedChainLocalManager`), run standalone after small edits (§2). Our code (`tree_alloc/`) only loads problems and the model, grades answers, saves trees, and does the analysis.

- §1 scope decisions
- §2 the code, and the edits to TreeRL's code
- §3 how to run, and the Task 1 results
- §4 Task 2 results
- §5 status

---

## 1. Scope decisions

| Task | Decision |
|---|---|
| **1. Harness** | TreeRL's code builds the trees; we log every tree (tokens, logprobs, fork points, step boundaries, rewards) to JSONL and compute all metrics offline. |
| **2. Replicate EPTree** | B0 (i.i.d.), B1 (random forks), B2 (EPTree); Figs 4, 5, 7, 8 and Table 2. **Deferred:** the R1-Distill "Wait"-token version of Fig. 7 (5–10× the cost); Fig. 7 is reported for Qwen2.5-Math. |

Two proposal-text issues noticed along the way:
- §4.2.2 repeats a paragraph (lines 167–175 = 176–184).
- The dataset differs between documents (MATH/MATH500 vs. Omni-MATH/MATH L4–5).

## 2. The code

**Tree building: TreeRL's code.** [`entropy_chain_local_manager.py`](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py) (the tree loop), [`tree_node.py`](../openrlhf/trainer/ppo_utils/tree_node.py) (`TreeNode`, fork masking, top-surprisal picking) and [`evaluation.py`](../openrlhf/trainer/ppo_utils/evaluation.py) (vLLM calls). The three methods:
- **b0** = TreeRL's (M,0,0,0), i.e. no forking;
- **b1** = random forking;
- **b2** = EPTree.

**Edits needed to run it standalone.** Three files, with the EPTree algorithm unchanged; `git diff openrlhf` shows everything.

| File | Change | Why |
|---|---|---|
| `evaluation.py` | `query_local_vllm_ids_with_logprobs`: with no Ray actor, call a local `vllm.LLM` using `TokensPrompt` | It assumed a Ray actor from their training job, and used a `prompt_token_ids=` argument that vLLM 0.30 removed |
| same | `check_result`: with no judge URLs, compare the last `\boxed{}` with the gold answer as normalized strings | Grading called LLM-judge servers on their lab network. It runs in a thread pool, where a symbolic checker can hang forever (it did, once). This grade only feeds TreeRL's RL rewards; every reported number is re-graded by our verifier |
| `parallel_mcts.py` | If the authors' config files (`/workspace/lurui/...`) don't exist, the judge/RM URL lists are empty | Importing it crashed otherwise; those URLs are only used by their MCTS |
| `entropy_chain_local_manager.py` | New `random_fork` flag: N uniformly random unmasked tokens per tree | The random-forking ablation (paper Table 2, our B1) isn't in the released code |

Also needed:
- **Packages:** `levenshtein`, `multiprocess`, `ipython`, `ray`, which their modules import.
- **Imports:** their modules load as plain modules (via their own `except: from tree_node import ...` fallback), so DeepSpeed and the training stack are never imported.
- **Stop tokens:** passed as token IDs; their Qwen config passes strings.

**Our code** (`mid_submission/`):

| File | Contents |
|---|---|
| [`run_treerl.py`](run_treerl.py) | Runs TreeRL's manager per problem, converts its trees to our format (`to_tree`), re-grades every leaf (`score`), appends one JSON line per tree. Resumes by problem ID; `--shard i --num_shards K` splits problems across parallel processes |
| [`task2.sh`](task2.sh) | All 11 Task 2 configs: 8 shards each, run by several processes per GPU. It waits for free GPU memory, retries a failed shard once, merges, then runs the reports |
| `tree_alloc/gen.py` | Loads the model in vLLM: raw logprobs, chat-template prompts as token IDs |
| `tree_alloc/data.py`, `verify.py` | Problem loaders; `\boxed{}` extraction and grading (`math_verify`, with a string fallback) |
| `tree_alloc/tree.py` | Tree log format (layout follows TreeRL's `TreeNode`): nodes, prefixes, "\n\n" step boundaries, JSON |
| `tree_alloc/task2_report.py` | Unbiased pass@k curve for i.i.d. sampling, Table 2, Figs 4/5/7/8 |
| `tree_alloc/metrics.py` | Per-tree metrics (summary table) and the paired bootstrap used for Table 2 |

**Deviations from the paper's setup:**
- Temperature is 1.0, not 1.2 (§3).
- The model is Qwen2.5-Math-1.5B-Instruct, not their 14B SFT model.
- Grading is rule-based, not an LLM judge.
- Surprisal is under raw π_θ.
- Fork masking is TreeRL's own: tokens after "answer"/"conclusion" are excluded, and there is no end-of-response mask.

## 3. How to run, and the Task 1 results

**Lab box (pkgpu2, `likhit@10.4.25.54`, 2× L40S 48 GB, driver 575 = CUDA ≤ 12.9, shared with other users).**
- Code: `~/likhit/tree_based_rl/TreeRL/mid_submission_likhit`.
- Environment: the uv env `~/likhit/.venv`, with `torch 2.13.0+cu129` and `vllm 0.30.0+cu129`. The default PyPI builds target CUDA 13, which this driver can't run.
- `gen.py` turns off FlashInfer's sampler, because the box's `nvcc` (12.0) can't compile it.

```bash
cd ~/likhit/tree_based_rl/TreeRL/mid_submission_likhit && source ~/likhit/.venv/bin/activate
nohup ./task2.sh > /tmp/task2.log 2>&1 &                 # Task 2 (resumable; GPUS=0 SLOTS=3 to limit)
python -m tree_alloc.run logs/task2/*.jsonl              # summary table
```

**Task 1 smoke test, TreeRL-code path** (20 MATH500 problems). All checks pass. They were run with a smoke-test script, removed after validation; its logs remain on pkgpu2 as `logs/smoke_*.jsonl`.

| Check | Result |
|---|---|
| Logprobs are raw π_θ | ✅ Re-scored with transformers: mean \|logged − raw\| = 0.0015–0.0025, vs. \|logged − processed (T = 1.0, top-p 0.95)\| = 0.0026–0.0053. 4/4 sequences are closer to raw. |
| "\n\n" segmentation | ✅ Mean 8.8 steps per response (median 8), median 38.5 tokens per step. |
| Verifier | ✅ Extracted answers match hand-checking on the sampled responses. |
| End to end | ✅ i.i.d. and EPTree trees built, graded and summarized (accuracy 0.713 on these easy problems). |
| Throughput | About 11k generated tok/s with 8 TreeRL processes across the two GPUs (their code handles one problem at a time, so it's sharded). |

**Temperature changed from 1.2 to 1.0.** At T = 1.2, 13% of samples collapse into random multilingual tokens, and those also have the highest surprisal, so EPTree would fork inside them. Results on 20 problems × 8 samples:

| T | acc | pass@8 | degenerate | mean tokens |
|---|---|---|---|---|
| 1.2 | 0.681 | 0.85 | 13.1% | 613 |
| **1.0** | 0.719 | **0.95** | 0% | 507 |
| 0.8 | 0.756 | 0.90 | 0% | 521 |

## 4. Task 2 results: TreeRL's code on 500 unfiltered Omni-MATH problems

**Setup:**
- Qwen2.5-Math-1.5B-Instruct, T = 1.0, top-p 0.95, max 3,072 response tokens, one seed.
- A random 500 from Omni-MATH (seed 0, `data/omni_math_500_seed0.jsonl` on pkgpu2), unfiltered, like the paper's Omni-MATH-500.
- Some Omni-MATH answers are free text, which a rule-based checker can't match (the paper used an LLM judge). That lowers absolute PassRate equally for every method.
- An earlier run on 243 accuracy-filtered OlympiadBench problems saturated (95% PassRate at 16 chains) and was dropped.

Full tables: [results/task2/results.md](results/task2/results.md). Figures: [Fig. 5](results/task2/fig5_passrate_vs_tokens.png), [Fig. 4](results/task2/fig4_responses_diversity_vs_tokens.png), [Fig. 7](results/task2/fig7_top_fork_tokens.png), [Fig. 8](results/task2/fig8_fork_position.png). Raw logs are on pkgpu2 in `logs/task2/` (1.4 GB).

**i.i.d. PassRate:** 25.9% at 1 chain, 40.0% at 8, **44.4% at 16**, 53.2% at 64 (paper, 14B model: 52.4% at 16, 67.4% at 64). 234 of the 500 problems are never solved in 64 samples.

**Table 2: entropy vs. random forking at the same (M,N,L,T).** Paired 95% bootstrap CIs over problems.

| (M,N,L,T) | Leaves | EPTree | Random | Δ [95% CI] | Distinct answers EPTree / random | Tokens EPTree / random | Paper Δ |
|---|---|---|---|---|---|---|---|
| (4,1,1,1) | 8 | 39.8% | 39.4% | +0.4 [−2.0, +2.8] | 3.65 / 3.63 | 4.6k / 4.6k | – |
| (4,3,1,1) | 16 | 43.0% | 42.2% | +0.8 [−1.6, +3.4] | 5.53 / 5.38 | 8.0k / 7.9k | – |
| (6,2,1,2) | 30 | 48.8% | 48.8% | 0.0 [−2.4, +2.4] | 8.41 / 8.07 | 14.5k / 14.2k | +2.1 (56.9 vs 54.8) |
| (8,4,2,2) | 136 | 56.4% | 57.2% | −0.8 [−3.2, +1.6] | 21.07 / 19.92 | 52.4k / 51.9k | +1.0 (71.0 vs 70.0) |

**Trees vs. i.i.d. at the same token cost (Fig. 5).** Tree PassRate vs. the i.i.d. pass@k curve interpolated to the same tokens per problem (from `results.md`; no CIs):

| Config | Tokens | EPTree | Random | i.i.d. at same tokens |
|---|---|---|---|---|
| (2,1,1,1), 4 leaves | 2.3k | 34.6% | – | 34.0% |
| (2,3,1,1), 8 leaves | 3.9k | 40.2% | – | 37.4% |
| (4,1,1,1), 8 leaves | 4.6k | 39.8% | 39.4% | 38.4% |
| (4,3,1,1), 16 leaves | 7.9–8.0k | 43.0% | 42.2% | 41.8–41.9% |
| (6,2,1,2), 30 leaves | 14.2–14.5k | 48.8% | 48.8% | 45.5–45.6% |
| (8,4,2,2), 136 leaves | 51.9–52.4k | 56.4% | 57.2% | 53.2% at 64 chains (47.3k tokens)* |

\* (8,4,2,2) costs more than the 64 i.i.d. chains sampled, so it can only be compared with pass@64, which used about 10% fewer tokens.

**Findings:**
1. **Trees beat i.i.d. sampling at the same token cost (Fig. 5 replicated).** Every tree config sits above the i.i.d. curve, by +0.4 to +3.3 points at matched tokens. (8,4,2,2) beats pass@64 by +3.2 / +4.0 with about 10% more tokens. Random forking gains as much as EPTree (+3.3 vs +3.2 at (6,2,1,2)), so the gain comes from **sharing prefixes**, not from where the forks are.
2. **Entropy vs. random forking (Table 2): no detectable difference.** Δ = +0.4, +0.8, 0.0, −0.8, and every CI includes 0. The paper's gaps (+2.1, +1.0) lie inside our CIs, so the data allow a small entropy advantage but give no evidence for one. The only consistent difference: EPTree yields slightly **more distinct answers** in all 4 configs (+0.02 to +1.15), i.e. it explores a little more widely.
3. **Fig. 8:** EPTree's fork positions are roughly uniform over the answer (mean 0.50), matching the paper, with a rise near the end because TreeRL has no end-of-response mask. Random forks lean later (mean 0.54), since later branches add more candidate positions.
4. **Fig. 7:** ` the`, ` \(`, ` \`, `,`, ` and`, ` a`, ` we`. This matches the paper, minus "Wait"/"But", which this model doesn't write. The mean surprisal at an EPTree fork is 4.95 nats (probability ≈ 0.7%).
5. **Only 17–48% of trees contain both correct and wrong answers** on this unfiltered set: 234 of the 500 problems are never solved in 64 i.i.d. samples, and 41 are always solved.

**Bottom line:** with TreeRL's own code and a 1.5B model, **tree sampling beats i.i.d. sampling at matched cost** (up to about +3 points), and fork positions are roughly uniform, as in the paper. **Entropy-guided forking does not measurably beat random forking** at 500 problems and one seed. Resolving a 1–2-point effect would need more seeds or problems.

## 5. Status

| Task | Status |
|---|---|
| 1. Harness | ✅ Done: TreeRL's tree builder, run standalone, plus our logging, grading and checks (§2–3) |
| 2. Replicate EPTree | ✅ Done with TreeRL's code on 500 unfiltered Omni-MATH problems (§4). Fig. 7 used Qwen2.5-Math-1.5B; the R1-Distill "Wait"-token analysis was deferred. |
| 3–6 (Phase I / II) | Out of scope for now. |

**Optional follow-ups:**
- Sample 128 i.i.d. chains so (8,4,2,2) has an exact token-matched baseline.
- Add seeds to resolve the entropy-vs-random question.
