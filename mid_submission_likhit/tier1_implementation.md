# Tier 1 Implementation: Task Review, Code, and Run Plan

Companion to `tier1_inference_plan.md` and the proposal (`ANLP_final_proposal (1).pdf`).

- Part 1 reviews the tasks against the proposal.
- Part 2 lists fixes to make before running.
- Part 3 describes the code in `tree_alloc/`.
- Part 4 gives the exact commands for the sweep.

---

## 1. Are the tasks relevant?

| Task | Verdict |
|---|---|
| **1. Harness** | Essential. Every metric is computed offline from the JSONL logs. |
| **2. Replicate EPTree** | **Keep:** B0–B2, Fig. 5, Table 2 and Fig. 4 as baselines, and Fig. 8 (fork positions), to compare against where P1 forks. **Defer:** Fig. 7 and R1-Distill. No proposal claim depends on "Wait" tokens, and they cost 5–10× more. Cut the (M,N,L,T) sweep to 2 configs per budget. |
| **3. Phase I (P1)** | Core. Needs the fixes in 2b and 2c. |
| **4. Phase II (P2)** | Core. Needs the fixes in 2a and 2e. |
| **5. Oracle** | **Most important.** It is the only task that directly tests the proposal's hypothesis (§6, "allocation quality"). Strengthen the metric as in 2d. |
| **6. Credit quality** | Make it required, not optional. It is the only Tier-1 link to §4.4, and it reuses the oracle rollouts, so it costs almost nothing extra. |

## 2. Fixes to make before running

**(a) Softmax over δ behaves almost like random allocation.**
- δ = 0 wherever a parent has only one child, and in a segment tree most nodes are like that.
- Those nodes still get weight exp(0) = 1 each. With about 100 nodes and at most about 22 with δ ≠ 0, most of the mass lands on δ = 0 nodes.
- As a result, P2-δ ≈ P2-rand, and Task 5 would reject δ for the wrong reason.

Fixes, all implemented:
- `--candidates branch_children` restricts candidates to children of branch points.
- **P2-spread** scores each branch point u directly by the std of V over its children. This is the "disagreement between child values" idea from proposal §8, and matches the fact that every sampled v sends its rollout to parent(v).
- Keep **P2-|δ|**. Plain softmax(δ/τ) gives the least weight to mistakes (negative δ).
- `summary` reports `delta_zero_frac`. On the fake model it is about 0.75.

**(b) σ_v = 0 almost everywhere early on.**
- In round 2 every node has |L(v)| = 1. Later, σ > 0 only for ancestors of existing branch points, so λσ pushes branching toward the root.
- μ_v is a suffix average: neighbouring nodes have nearly the same μ, and late nodes have short, low-surprisal suffixes.
- Prediction: P1 forks early. Check with `fork_rel_pos`, the Fig. 8 statistic.
- `--min_suffix` (default 32 tokens) keeps near-leaf states out.

**(c) Keep Phase I sequential (Algorithm 1).**
- Top-B branching per round is not needed for speed. All problems advance in lockstep, so each round is **one batched vLLM call across every problem**.
- `--per_round 1` is Algorithm 1; `--per_round B` is the ablation.
- Phase II computes p once (Algorithm 1, line 15), so all k rollouts go out in one call.

**(d) p̂(1−p̂) is not "consequentiality".**
- It decomposes as p(1−p) = Var_a[Q(u,a)] + E_a[Q(1−Q)]. A state where every continuation is a coin flip maximizes p(1−p) but involves no consequential decision.
- The oracle therefore also measures **`var_between`**: sample K next segments from u, run M rollouts after each, and take the variance of the per-segment success rate Q̂.
- All P2 variants share the same P1 tree, so each variant is scored by its **expected** oracle value Σ_u p(u)·metric(u). This avoids generating the k rollouts for Task 5 and removes their sampling noise.

**(e) Missing baselines and metrics.**
- **P1-full (k = 0):** spend the whole budget n on Phase I. Without it, "reserving k for Phase II helps" is untested.
- **Report both budget axes:** leaves (the proposal's n) and generated tokens (the plan's axis). A fork from a deep state costs fewer tokens than a full chain.
- **`mixed` / `mixed_branch_frac`:** whether a tree or branch point has both correct and incorrect outcomes. All-correct or all-wrong trees give zero advantage.

**(f) Proposal text.**
- §4.2.2 repeats a paragraph (lines 167–175 = 176–184).
- §4.4 never defines the advantage. Commit to TreeRL's (G_A + L_A)/√|L(s)|, implemented as `segments.treerl_step_rewards`.
- The dataset differs between documents (MATH/MATH500 vs Omni-MATH/MATH L4–5). Pick one.

---

## 3. What is implemented (`mid_submission/tree_alloc/`)

This is standalone code. It does not import `openrlhf`, which pulls in Ray and DeepSpeed and needs remote judge servers. The core runs on the Python ≥3.9 standard library; vLLM and `math_verify` are imported only on the GPU machine.

| File | Contents |
|---|---|
| `tree.py` | `Node` (one generation call: `parent`, `fork_idx`, `offset`, `token_ids`, `logprobs`, `seg_ends`, `boxed_at`, `reward`) and `Tree` (`response`, `prefix_ids`, `locate`, `boundaries`, `new_tokens`, JSON). The layout follows TreeRL's `TreeNode`. Forking always uses token ids. |
| `segments.py` | `SegTree`: a trie of "\n\n" segments over all leaves. Works for EPTree trees too, where a mid-segment fork becomes a sibling. Provides V, δ, H_{ℓ,v}, μ_v, σ_v, eligibility, fork points, and the TreeRL step reward. |
| `gen.py` | `VLLMGenerator`: token-id prompts, `enable_prefix_caching`, the sampled-token logprob, and `logprobs_mode="raw_logprobs"` when supported. `FakeGenerator`: deterministic toy model for tests. |
| `methods.py` | B0, B1/B2 (EPTree: top-N per initial chain, M + L·M·N·T leaves), P1 (`--score mu / sigma / mu_sigma / random`, `--lam`, `--per_round`), P2 (`rand / unif / delta / absdelta / spread` × `--candidates all / branch_children`). |
| `oracle.py` | p̂ from R rollouts, `var_between` from K next steps × M rollouts, and expected-value scoring. |
| `metrics.py` | Per-tree metrics (acc, pass_any, mixed, distinct answers, branch points, δ-sparsity, fork positions and tokens), Spearman correlation, paired bootstrap. |
| `run.py` | CLI: `gen`, `p2`, `oracle`, `summary`, `report`. Writes JSONL and resumes by problem id. |
| `tests/` | 18 unit tests plus an end-to-end CLI run on the fake backend. Run them with `python3 -m unittest discover -s tests`. |

**Task → output:**

| Task | Where to get it |
|---|---|
| Fig. 4 | `summary`: `n_leaves`, `new_tokens`, `distinct_answers` |
| Fig. 5 | `acc` / `pass_any` vs `new_tokens` across configs |
| Table 2 | `b1` vs `b2` at the same (M,N,L,T) |
| Fig. 8 | `fork_rel_pos` |
| Fig. 7 | `fork_tokens` (decode with the tokenizer) |
| Task 5 and δ reliability | `report`: expected `var_total` / `var_between` per variant with bootstrap CIs vs P2-rand; Spearman of \|δ\| and spread against the oracle |
| Task 6 | `report`: \|V − p̂\| by leaf count, and for each final-tree log passed with `--trees` |

**Where this deviates from TreeRL's code (state these in the report):**
- Surprisal is measured under raw π_θ.
- Positions already forked are masked; TreeRL can re-pick them.
- TreeRL masks everything after any token containing "answer". This code uses the last 10% of tokens and `\boxed` instead.
- The step reward follows the paper's formula. TreeRL's code applies leave-one-out normalization and √ weighting before taking differences.

---

## 4. How to run (GPU machine)

**Lab box (`likhit@10.4.25.54`, 2× L40S 48 GB, driver 575 = CUDA ≤ 12.9).**
- The code lives at `~/likhit/tree_based_rl/TreeRL/mid_submission`. The uv env is `~/likhit/.venv`.
- The env has `torch 2.13.0+cu129` and `vllm 0.30.0+cu129`. Both default builds on PyPI target CUDA 13, which this driver cannot run.
- To reinstall vLLM: `uv pip install --python ~/likhit/.venv/bin/python "vllm==0.30.0+cu129" --extra-index-url https://wheels.vllm.ai/0.30.0/cu129 --torch-backend=cu129 --index-strategy unsafe-best-match`
- Always `source ~/likhit/.venv/bin/activate` first.
- `gen.py` turns off FlashInfer's sampler, because it compiles kernels at runtime and the box's `nvcc` is 12.0.
- Check `nvidia-smi` before starting: GPU 0 is shared with other users.

```bash
cd ~/likhit/tree_based_rl/TreeRL/mid_submission && source ~/likhit/.venv/bin/activate
python -m unittest discover -s tests           # sanity check, no GPU needed
export CUDA_VISIBLE_DEVICES=1                  # a free GPU; one config per GPU
R="python -m tree_alloc.run"                   # model: Qwen/Qwen2.5-Math-1.5B-Instruct, 4k context
```

**Smoke test results (2026-09-25, `./smoke.sh`, 20 MATH500 problems, GPU 1). Task 1 checks pass:**

| Check | Result |
|---|---|
| Logprobs are raw π_θ | ✅ Re-scored with transformers: mean \|vLLM − raw\| = 0.002–0.005 vs \|vLLM − logits/T\| = 0.005–0.018. 4/4 sequences are closer to raw. |
| "\n\n" segmentation | ✅ Mean 10.2 segments per response (median 8), median 39 tokens per segment. Boundaries fall at step headers and sentence breaks. |
| Verifier | ✅ Agrees on the 30 samples checked by hand. Every miss was a genuinely wrong answer or a missing `\boxed`. |
| All methods run end to end | ✅ B0, B2 and P1 all produce scored trees and `summary` tables. |
| Throughput | About 4–6k generated tok/s at 160 concurrent sequences (the 1.5B model, one L40S). It should rise with larger batches. |

**Temperature changed from 1.2 to 1.0 (the new default).** At T = 1.2, **13% of samples collapse** into random multilingual tokens. These also have the highest surprisal, so EPTree and P1 would branch into garbage. Results from 20 problems × 8 samples:

| T | acc | pass@8 | degenerate | mean tokens |
|---|---|---|---|---|
| 1.2 | 0.681 | 0.85 | 13.1% | 613 |
| **1.0** | 0.719 | **0.95** | 0% | 507 |
| 0.8 | 0.756 | 0.90 | 0% | 521 |

State this as a deviation from TreeRL, which used 1.2 on 9B–14B models. MATH500 is too easy for this model (pass@8 = 0.95), which confirms the plan to use harder problems plus the 0.1–0.9 accuracy filter.

**Step 0. Smoke test (20 problems).** Before any sweep, check by hand:
- Returned logprobs are raw: the startup log says `logprobs_mode`.
- "\n\n" segmentation looks right on real outputs. Print `Tree.boundaries` on a few trees.
- The verifier agrees with you on about 30 outputs.
- Measure tokens per second.

```bash
$R gen --method b0 --n 4 --data ../datasets/eval/MATH500.jsonl --limit 20 --out logs/smoke.jsonl
$R summary logs/smoke.jsonl
```

**Step 1. Pick problems.**
- Keep problems whose accuracy is in 0.1–0.9, using a *separate* B0 run. Don't reuse it as the baseline, because the selection would bias it.
- Omni-MATH has to be downloaded. `../datasets/eval/olympiad_bench.jsonl` (675 problems) also works.

```bash
$R gen --method b0 --n 16 --seed 100 --data $DATA --limit 800 --out logs/select_b0.jsonl
SEL="--data $DATA --select_from logs/select_b0.jsonl --acc_min 0.1 --acc_max 0.9 --limit 250"
```

*Done:* `logs/select_b0_olympiad.jsonl` covers all 675 OlympiadBench problems, and 243 of them fall in the 0.1–0.9 band. That run is used only for selection.

**Step 2a. Task 2: B0/B1/B2 sweep (done).** Run `nohup ./task2.sh > /tmp/likhit_task2.log 2>&1 &` (see [task2.sh](task2.sh)), then `python -m tree_alloc.task2_report --logs logs/task2 --out results/task2`. Results are in §5.

**Step 2b. Task 3: Phase I table at n = 16 leaves (not started).** Use the same `SEL` as `task2.sh`, then compare against the Task 2 logs.

```bash
for LAM in 0 0.5 1 2; do
  $R gen --method p1 --budget 16 --lam $LAM       $SEL --out logs/p1full_lam$LAM.jsonl   # P1-full (k=0)
done
$R gen --method p1 --budget 16 --score sigma      $SEL --out logs/p1full_sigma.jsonl
$R gen --method p1 --budget 16 --score random     $SEL --out logs/p1full_random.jsonl    # segment-level random
$R gen --method p1 --budget 16 --per_round 3      $SEL --out logs/p1full_batched.jsonl   # batching ablation
$R summary logs/b*_*.jsonl logs/p1full_*.jsonl
```

**Step 3. Phase II: k/n = 0.25 (12 + 4) and 0.5 (8 + 8).**

```bash
$R gen --method p1 --budget 12 --lam 1 $SEL --out logs/p1_12.jsonl
$R gen --method p1 --budget 8  --lam 1 $SEL --out logs/p1_8.jsonl
for V in rand unif delta absdelta spread; do
  $R p2 --from logs/p1_12.jsonl --variant $V --k 4 --out logs/p2_12_$V.jsonl
  $R p2 --from logs/p1_8.jsonl  --variant $V --k 8 --out logs/p2_8_$V.jsonl
done
$R p2 --from logs/p1_12.jsonl --variant absdelta --candidates branch_children --k 4 --out logs/p2_12_absdelta_bc.jsonl
$R summary logs/p1full_lam1.jsonl logs/p2_12_*.jsonl logs/p2_8_*.jsonl   # vs P1-full at the same n
```

**Step 4. Oracle (the Tier 2 decision gate) and Task 6.**
- About 40 problems × up to 30 states × (32 + 8·4) rollouts.
- Add τ values with `--variants "delta:0.25:all,delta:1:all,..."`.

```bash
$R oracle --from logs/p1_12.jsonl --limit 40 --max_states 30 --R 32 --K 8 --Mq 4 --out logs/oracle_12.jsonl
$R report --p1 logs/p1_12.jsonl --oracle logs/oracle_12.jsonl --trees logs/p2_12_*.jsonl
```

**Decision gate for Tier 2:** proceed only if P2-|δ|, P2-δ or P2-spread beats P2-rand on **`var_between`**, with a bootstrap 95% CI above 0. If none does, revise the consequentiality signal first.

---

## 5. Task 2 results: EPTree replication (2026-09-25)

**Setup.**
- Model: Qwen2.5-Math-1.5B-Instruct, T = 1.0, top-p 0.95, max 3072 response tokens.
- Problems: **243 OlympiadBench problems** with B0 accuracy between 0.1 and 0.9 (from the separate selection run).
- One seed. Fork positions in the last 10% of a response and after `\boxed` are excluded.
- Compute: about 1 h 50 min on one L40S.
- Full tables: [results/task2/results.md](results/task2/results.md). Raw logs are on the server under `mid_submission/logs/task2/` (about 1.3 GB, not copied locally).

| Figure | File |
|---|---|
| Fig. 5: PassRate vs. tokens | [fig5_passrate_vs_tokens.png](results/task2/fig5_passrate_vs_tokens.png) |
| Fig. 4: responses and distinct answers vs. tokens | [fig4_responses_diversity_vs_tokens.png](results/task2/fig4_responses_diversity_vs_tokens.png) |
| Fig. 7: top-10 forking tokens | [fig7_top_fork_tokens.png](results/task2/fig7_top_fork_tokens.png) |
| Fig. 8: where forks happen | [fig8_fork_position.png](results/task2/fig8_fork_position.png) |

**Main table.** Leaves = M + L·M·N·T. The i.i.d. column is the unbiased pass@k curve from 64 chains per problem, interpolated to the same token budget.

| Method | (M,N,L,T) | Leaves | Tokens/problem | PassRate | i.i.d. at same tokens | Mean acc | Distinct answers |
|---|---|---|---|---|---|---|---|
| i.i.d. | k = 16 | 16 | 13,096 | 95.0% | – | 0.458 | 6.13 |
| EPTree | (4,1,3,1) | 16 | 8,773 | 90.1% | 91.3% | 0.443 | 5.65 |
| EPTree | (4,3,1,1) | 16 | 9,454 | 93.8% | 92.0% | 0.438 | 5.86 |
| Random | (4,3,1,1) | 16 | 9,024 | 92.2% | 91.5% | 0.463 | 5.47 |
| EPTree | (8,1,1,1) | 16 | 10,772 | 94.2% | 93.2% | 0.444 | 5.95 |
| EPTree | (6,2,1,2) | 30 | 17,370 | 97.9% | 96.3% | 0.451 | 8.70 |
| Random | (6,2,1,2) | 30 | 16,377 | 97.9% | 96.1% | 0.451 | 8.05 |
| i.i.d. | k = 64 | 64 | 52,489 | 99.6% | – | 0.458 | 13.31 |
| EPTree | (8,1,7,1) | 64 | 29,444 | 99.6% | 98.5% | 0.421 | 12.93 |
| EPTree | (8,7,1,1) | 64 | 36,000 | 98.4% | 98.8% | 0.447 | 13.05 |
| EPTree | (16,3,1,1) | 64 | 38,201 | 100.0% | 99.0% | 0.440 | 13.56 |
| Random | (16,3,1,1) | 64 | 36,141 | 99.2% | 98.9% | 0.457 | 12.80 |

**Table 2: entropy-guided vs. random forking at the same (M,N,L,T).** CIs are paired bootstrap 95% intervals over problems.

| (M,N,L,T) | PassRate EPTree | PassRate random | Δ [95% CI] | Distinct EPTree / random | Acc EPTree / random |
|---|---|---|---|---|---|
| (4,3,1,1) | 93.8% | 92.2% | +1.6 [−2.1, +5.3] | 5.86 / 5.47 | 0.438 / 0.463 |
| (6,2,1,2) | 97.9% | 97.9% | +0.0 [−2.5, +2.1] | 8.70 / 8.05 | 0.451 / 0.451 |
| (16,3,1,1) | 100.0% | 99.2% | +0.8 [+0.0, +2.1] | 13.56 / 12.80 | 0.440 / 0.457 |

**Findings.**
1. **The token savings from trees replicate (Fig. 4).** Branching reuses shared prefixes. The 16-leaf trees cost 8.8–10.8k tokens versus 13.1k for 16 i.i.d. chains (18–33% fewer). The 64-leaf trees cost 29–38k versus 52.5k (27–44% fewer). The deep config (8,1,7,1) is the cheapest.
2. **EPTree beats i.i.d. at matched tokens, but only slightly (Fig. 5).** It is +1 to +1.8 PassRate points in 5 of 7 configs, and slightly below in (4,1,3,1) at −1.2 and (8,7,1,1) at −0.4. The direction matches the paper, but the effect is small because PassRate saturates: i.i.d. pass@16 is already 95%.
3. **Entropy vs. random forking (Table 2) is not a clear win at this scale.** PassRate differences are within noise (only (16,3,1,1) is borderline). Entropy forking consistently gives **more distinct answers** (+0.4 to +0.8) and **lower per-leaf accuracy** (up to −0.025): it explores more, and the extra branches are more often wrong. It also uses 5–6% more tokens than random at the same config, consistent with its earlier forks (mean relative fork position 0.43 vs. 0.45), which leave longer continuations to generate. The result is **inconclusive rather than contradictory**. The paper's own Table 2 gap is small: +2.1 at (6,2,1,2) and +1.0 at (8,4,2,2), from one run on Omni-MATH-500 with no CIs. That is the same size as our unsaturated point estimates (+1.6, +0.8), and smaller than a 243-problem set can resolve. The paper also found random forking used *more* tokens than EPTree (24.2k vs 22.3k); here it used fewer.
4. **Forking tokens (Fig. 7):** mostly the start of inline math (`␣\(`, `␣\`), function words (`␣the`, `␣we`, `␣and`) and punctuation (`,`, `.`, `:\n`). These are points where the model picks how to phrase or start the next step. The distribution is flat: the top 10 cover only about 13% of forks. Mean surprisal at an EPTree fork is **3.75 nats** (p ≈ 0.02), versus 0.14 for random forks.
5. **Where forks happen (Fig. 8):** random forks are spread evenly. EPTree forks cluster at the **start** of the response (0–15%) and just before the **end** (80–90%, right before the 10% tail mask, which is why the curve drops to zero after 0.9).
6. **Almost every tree has both correct and incorrect leaves** (86–100%, because of the 0.1–0.9 selection band), so the trees give a usable RL signal.

### 5.1 Add-on analyses on the filtered set

Full tables: [results/task2/extra.md](results/task2/extra.md). Source: [task2_extra.py](tree_alloc/task2_extra.py). Everything is computed offline from the same logs.

**A. Tree vs. i.i.d., paired per problem.** i.i.d. cost for each problem is k × that problem's mean chain length; values are unbiased pass@k, with 95% bootstrap CIs.

| Config | Δ vs. i.i.d., same tokens | Δ vs. i.i.d., same leaves |
|---|---|---|
| EPTree (4,3,1,1) / (8,1,1,1) | +1.8 [−1.0, +4.3] / +0.8 [−1.7, +3.4] | −1.2 / −0.8 |
| EPTree (4,1,3,1), deep | −1.2 [−4.7, +1.9] | −4.9 [−8.6, −1.6] |
| EPTree (6,2,1,2) / random | +1.5 [−0.1, +3.0] / +1.7 [−0.2, +3.8] | −0.1 / −0.1 |
| EPTree (16,3,1,1) / random | **+1.0 [+0.3, +2.1]** / +0.1 [−1.4, +1.5] | +0.4 / −0.4 |
| EPTree (8,1,7,1) / (8,7,1,1) | +1.2 [+0.0, +2.6] / −0.6 [−2.2, +0.7] | +0.0 / −1.2 |

At the same token budget, trees do as well as or slightly better than i.i.d. sampling: +1 to +2 points, significant only for (16,3,1,1). At the same number of leaves they do as well or worse, because leaves that share prefixes are correlated. **Any tree advantage comes from the token savings, not from better leaves.**

**B. Sibling disagreement.**
- A fork's outcome differs from the original it branched off in **20–26%** of cases. Two independent chains differ in **34.4%**, since forks share a prefix with the original.
- Raw EPTree − random differences are +0.6, +4.3 and +2.0 points (the last two significant). **Most of this is fork position:** earlier forks share less prefix, and EPTree forks earlier.
- Reweighting random forks to EPTree's position mix gives **23.6% vs. EPTree's 24.3%** (raw random: 21.8%).
- By position band: EPTree is about 1 point *lower* in the first 40% of the response and 2–3 points *higher* after 40%.
- Disagreement falls steeply with position: **32% in the first fifth → 8–10% in the last fifth.**

**C. Continuation length.** A fork's new branch is as long as the part of the original it replaces (median ratio 1.00–1.01), and under 0.5% of forks get truncated. EPTree's extra ~5% of tokens come entirely from forking earlier: its forks cut off 505–519 remaining tokens, versus 451–480 for random forks.

**D. Estimating V(root) against the policy's pass@1 of 0.458.**
- The **leaf mean** (the proposal's V(v), eq. 2) is **biased low under EPTree**: −0.007 to −0.037, significant in 4 of 7 configs. The worst case is the deep (8,1,7,1) at −0.037 [−0.050, −0.025].
- Under **random forking it is unbiased**: −0.007 to +0.005, with no config significant.
- The child mean reduces the bias of the deep config (−0.037 → −0.013), but not consistently elsewhere.
- The roots-only estimate is unbiased by construction. One of its 10 CIs excludes 0, which fits chance at 95%.
- Despite the bias, the leaf mean still has the **lowest error at the root** (MAE 0.065–0.121, vs. 0.088–0.170 for roots-only), because extra leaves cut variance more than the bias costs.

**What this means for the proposal:**
1. **Any fork-selection rule that looks at the chain biases the leaf mean.** EPTree does, and P1's continuation-cross-entropy rule will too. Task 6 should report the roots-only and child-mean estimators alongside the leaf mean.
2. **Position confounds consequentiality.** Early forks flip outcomes about 3× more often than late ones. P2 variants should be compared against a *position-matched* random baseline, not only uniform random.
3. **Token surprisal is a weak consequentiality signal once position is controlled.** That raises the bar for the proposal's continuation-cross-entropy and δ signals, and makes Task 5's oracle the right test.

**Implications for Task 3 (not started):**
- PassRate is already saturated at 16 leaves on these problems. For the Phase I comparison, also report metrics that don't saturate: pass@k at 8 leaves, distinct answers, `mixed_branch_frac`.
- Alternatively, use a harder subset (B0 accuracy 0.1–0.5).
- Single-seed differences of 1–2 points are within noise at 243 problems, so use paired bootstrap CIs throughout.

## 6. Status

| Task | Status |
|---|---|
| 1. Harness | ✅ Done and checked on the real model (§4) |
| 2. Replicate EPTree | ✅ Done (§5). Fig. 7 used Qwen2.5-Math-1.5B; the R1-Distill "Wait"-token analysis was deferred as planned. |
| 3–6 | Code written and unit-tested. Not run yet. |

**Still open:**
- Download Omni-MATH (OlympiadBench was used instead).
- Choose the τ grid for P2 before Task 4.
