# Tier 1 (Inference Only): Replicate EPTree + Evaluate Credit-Aware Rollout Allocation

This tier reproduces TreeRL's sampling results (Table 2, Figs 4, 5, 7, 8) and tests our Phase I heuristic and Phase II allocation against them.

**Everything in this tier is inference only.** It uses generation, per-token logprobs from that same generation, answer verification, and offline statistics. There are no gradient updates and no optimizer state.

---

## 1. Setup

### Model
- **Default:** `Qwen2.5-Math-1.5B-Instruct`, which is cheap and gives short outputs.
- **Optional:** `DeepSeek-R1-Distill-Qwen-1.5B`, only for the "Wait"-token analysis (Fig. 7). Its outputs run 8–16k tokens and cost roughly 5–10× more, so cap `max_tokens`.
- A 7B model fits on 48GB with vLLM if needed.

### Sampling
- Temperature 1.2, top-p 0.95 (same as the paper).

### Framework: vLLM with a custom fork loop
- Generate with `logprobs=1`, which gives the sampled-token logprob $\log p_t$ at every position.
- **Fork using token IDs:** `prompt_token_ids = prompt_ids + response_ids[:t]`.
  - Never decode the prefix and re-tokenize it, because token boundaries can shift at the fork point.
  - Do not append EOS or chat-template closing tokens after the prefix.
- Set `enable_prefix_caching=True`.
- **Budget accounting:** count only newly generated tokens.

### Definitions
- **Uncertainty (EPTree):** surprisal of the sampled token, $-\log \pi_\theta(y_t \mid x, y_{<t})$. This is not full-distribution entropy; state this in the report.
- **End-of-sequence masking:** exclude fork points in the last 10% of tokens or after `\boxed{`. Report the choice.
- **Segmentation (our method):** split responses on `\n\n`. A node is a segment, and forks happen at segment boundaries.

### Dataset
- 200–300 problems from Omni-MATH or MATH level 4–5.
- Avoid easy MATH500 problems, where PassRate saturates for 1.5B models at 16+ samples.
- Use a fixed subset and fixed seeds across all methods.

### Verifier
- `math_verify` or an equivalent boxed-answer checker. Reward is 1 or 0 per leaf.

---

## 2. Methods

All methods are compared at **matched generated-token budgets**.

| ID | Method |
|---|---|
| B0 | i.i.d. multi-chain |
| B1 | Random forking, same (M, N, L, T) as EPTree |
| B2 | EPTree: top-N token surprisal |
| P1-μ | Ours, Phase I: mean continuation CE only |
| P1-σ | Ours, Phase I: std of continuation CE only |
| P1 | Ours, Phase I: $\mu_v + \lambda\sigma_v$ |
| P2-rand | P1 tree + k rollouts at random nodes |
| P2-unif | P1 tree + k rollouts spread uniformly over branch points |
| P2-δ | P1 tree + k rollouts via $\text{softmax}(\delta/\tau)$ (as proposed) |
| P2-\|δ\| | P1 tree + k rollouts via $\text{softmax}(\lvert\delta\rvert/\tau)$ |

MCTS is dropped because it's not relevant to our proposal.

---

## 3. Tasks

### Task 1: Harness
- Build the vLLM fork loop, the tree data structure (nodes with token spans, parent, children, and leaf rewards), token accounting, and the verifier.
- Write **one JSONL log per tree** containing:
  - every token's logprob,
  - the fork points,
  - the segment boundaries,
  - the leaf rewards.
- All later analyses read from these logs.

### Task 2: Replicate EPTree (B0–B2)
- Sweep (M, N, L, T) configurations around 16- and 64-chain budgets (paper Table 4).
- Plot PassRate vs. generated tokens (Fig. 5).
- Plot number of responses and number of distinct final answers vs. tokens. This is Fig. 4 plus a real diversity measure.
- Run the entropy vs. random forking ablation at fixed (M, N, L, T) (Table 2).
- Plot the top-10 forking-token histogram (Fig. 7) and the relative fork-position histogram (Fig. 8).

### Task 3: Our Phase I heuristic (P1 variants)
- Compute continuation cross-entropy from the logprobs already logged. No extra forward passes are needed:

$$H_{\ell,v} = -\frac{1}{L_{\ell,v}} \sum_{t \in \text{suffix}_{\ell,v}} \log p_t$$

$$\mu_v = \frac{1}{|L(v)|}\sum_{\ell \in L(v)} H_{\ell,v}, \qquad \sigma_v^2 = \frac{1}{|L(v)|}\sum_{\ell \in L(v)} (H_{\ell,v} - \mu_v)^2$$

$$S_{\text{explore}}(v) = \mu_v + \lambda \sigma_v$$

- **Batch the branching:** pick the top-B nodes per round instead of one node at a time. This avoids $n-k-1$ sequential generation rounds.
- Sweep $\lambda \in \{0, 0.5, 1, 2\}$.
- Compare against B0–B2 on PassRate vs. tokens.

### Task 4: Phase II allocation (P2 variants)
- Take the P1 trees and compute:

$$V(v) = \frac{1}{|L(v)|}\sum_{\ell \in L(v)} R_\ell, \qquad \delta(v) = V(v) - V(\text{parent}(v))$$

- Allocate k rollouts to $\text{parent}(v)$ under each P2 variant.
- Sweep $k/n \in \{0.25, 0.5\}$ and $\tau$.
- Log the fraction of nodes with $\delta = 0$ exactly.
  - This is structurally expected: if $\text{parent}(v)$ has only one child, then $L(v) = L(\text{parent}(v))$, so $\delta(v) = 0$.
  - The question is whether Phase II has enough candidate nodes.

### Task 5: Oracle experiment (tests the main claim)
- On a subset of 30–50 problems, generate 32 rollouts from each candidate parent state to estimate its true success rate $\hat{p}_v$.
- **Metric:** mean outcome variance $\hat{p}_v(1 - \hat{p}_v)$ at the states selected by each P2 variant. Higher means more informative counterfactual evidence.
- Also report the rank correlation between $\lvert\delta\rvert$ (from the small tree) and the oracle variance. This measures how reliable $\delta$ is when each node has only 1–3 leaves.

### Task 6 (optional): Credit-signal quality
- For the final trees from each method, compute TreeRL's step reward:

$$R(s) = \frac{G_A(s) + L_A(s)}{\sqrt{|L(s)|}}, \quad G_A(s) = V(s) - V(\text{root}), \quad L_A(s) = V(s) - V(p(s))$$

- On the oracle subset, compare each method's node values against the oracle values: compute the error of $V(s)$ against $\hat{p}_s$, per method.

---

## 4. Outputs

- Figs 4, 5, 7 and 8 and Table 2, replicated.
- **Phase I table:** PassRate and tokens for B0–B2 and the P1 variants.
- **Phase II table:** oracle variance at selected states, and the rank correlation, per P2 variant.
- **$\delta$ sparsity and noise analysis.** This informs the choice between $\delta$ and $\lvert\delta\rvert$ before any training.

## 5. Decision gate for Tier 2 (RL training)

Proceed to training with P2-δ or P2-|δ| **only if one of them beats P2-rand on oracle variance (Task 5)**. Otherwise, revise the consequentiality signal first.

**What Tier 1 cannot show:** whether better allocation produces a better-trained policy. That requires Tier 2. Tier 1 establishes the precondition: more informative evidence (Task 5) and more accurate node values (Task 6).

---

## 6. Rough compute estimate

These are estimates for a 1.5B model on one 48GB GPU. Measure actual throughput on a small run first, then scale the sweep to fit.

| Item | Tokens | Time (at ~5–15k tok/s batched) |
|---|---|---|
| One config: 64-chain budget, 250 problems, ~1k tok/answer | ~16M | ~20–60 min |
| Oracle: 40 problems × ~10 states × 32 rollouts × ~500 tok | ~6.4M | ~10–25 min |
| Full tier, depending on sweep size | — | ~10–30 GPU-hours |

R1-Distill costs roughly 5–10× more because of its long outputs, so restrict it to the Fig. 7 analysis.
