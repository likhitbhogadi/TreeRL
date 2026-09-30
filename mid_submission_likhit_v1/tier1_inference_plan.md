# Tier 1 (Inference Only): Replicate EPTree

This tier reproduces TreeRL's sampling results (Table 2, Figs 4, 5, 7, 8). Our Phase I / Phase II methods (former Tasks 3–6) are out of scope for now.

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

---

## 4. Outputs

- Figs 4, 5, 7 and 8 and Table 2, replicated.

## 5. Rough compute estimate

These are estimates for a 1.5B model on one 48GB GPU. Measure actual throughput on a small run first, then scale the sweep to fit.

| Item | Tokens | Time (at ~5–15k tok/s batched) |
|---|---|---|
| One config: 64-chain budget, 250 problems, ~1k tok/answer | ~16M | ~20–60 min |
| Full Task 2 sweep, depending on size | — | ~2–5 GPU-hours |

R1-Distill costs roughly 5–10× more because of its long outputs, so restrict it to the Fig. 7 analysis.
