This is a well-executed Task 2, and the results tell a pretty clear (if slightly deflating) story. Let me walk through each figure against the paper, then say what it means for Tasks 3–6.

## 1. Token savings (Fig. 4) — replicates cleanly

Your tree configs land on or slightly above the i.i.d. token-savings curve: 16 leaves in 8.8–10.8k tokens vs. 13.1k for 16 i.i.d. chains (18–33% cheaper), 64 leaves in 29–38k vs. 52.5k (27–44% cheaper). This matches the paper's core structural claim — shared prefixes really do save generation cost — and it's not sensitive to model scale, since it's mostly bookkeeping (how much of a response is shared vs. new). No surprises here, good sanity check that the harness is accounting tokens correctly.

## 2. PassRate vs. tokens (Fig. 5) — direction matches, effect size doesn't

You do see EPTree beating the i.i.d. pass@k curve at matched tokens in 5/7 configs, by roughly 1–1.8 points — same direction as the paper's ~3-point PassRate advantage on Omni-MATH-500 with Qwen-2.5-14B-SFT. But your effect is smaller and less consistent (two configs, (4,1,3,1) and (8,7,1,1), are actually *below* the i.i.d. curve).

The likely reason is visible in your own baseline table: **i.i.d. pass@16 is already 95.0%** on this problem set. You're measuring a method's ability to find "at least one correct answer" on a metric that's already 95% saturated at the budget where you're comparing — there's only 5 points of headroom left to fight over, so a 1–2 point swing is basically the whole game, and noise at 243 problems dominates. The paper had more headroom because Omni-MATH-500 is harder for their base models. You partially compensated by selecting problems in the 0.1–0.9 accuracy band, but pass@16 (aggregated across the *set*) still saturates fast because "at least one correct in 16 tries" is a weak bar even for a 45%-accuracy-per-sample model.

## 3. Table 2: entropy vs. random forking — this is the important negative result

This is where your finding genuinely diverges from the paper. TreeRL reports entropy-guided forking beating random forking by ~2 PassRate points fairly consistently (Table 2 in their paper: 56.9 vs 54.8, 71.0 vs 70.0). Your CIs all include zero:

| (M,N,L,T) | Δ PassRate (B2−B1) | 95% CI |
|---|---|---|
| (4,3,1,1) | +1.6 | [−2.1, +5.3] |
| (6,2,1,2) | +0.0 | [−2.5, +2.1] |
| (16,3,1,1) | +0.8 | [+0.0, +2.1] |

So at this scale (1.5B, single seed, 243 problems), **entropy-guided forking is not distinguishable from random forking on PassRate.** But look at what *is* consistently different between B1 and B2:

- **Mean surprisal at the fork token: 3.75 nats (EPTree) vs. 0.14 nats (random).** This is the important confirmation — your selection mechanism is doing exactly what it's supposed to do; it's finding genuinely rare, high-uncertainty tokens, by a huge margin. The mechanism isn't broken.
- **Distinct answers is consistently higher for EPTree** (e.g., 13.56 vs 12.80 at (16,3,1,1)) — forking at genuinely uncertain points does diversify the answer set.
- **Per-leaf accuracy is consistently lower for EPTree** (e.g., 0.440 vs 0.457) — the extra diversity comes partly from branches that go wrong, not just branches that find new correct paths.
- **EPTree uses slightly more tokens** at matched (M,N,L,T) (e.g., 38,201 vs 36,141), consistent with earlier mean fork positions (0.43 vs 0.45) leaving longer continuations.

Put together: **you're correctly finding high-uncertainty tokens, and they do diversify the answer set, but at 1.5B scale that diversification doesn't reliably convert into more *correct* answers** — it's a wash between "found a new correct path" and "found a new wrong path." This is a genuinely useful negative result, not a bug, and it directly motivates your Phase I proposal: single-token surprisal is a noisy proxy for "this fork will matter," and your $\mu_v + \lambda\sigma_v$ signal (aggregated over *leaves*, i.e., does branching here actually change the *outcome distribution*, not just the *token distribution*) is a more direct attempt to measure the thing you actually care about. Task 2 essentially validates the premise for Task 3 rather than closing the book on EPTree.

## 4. Fig. 7 — forking tokens are qualitatively similar to the paper

Your top tokens (`\(`, "the", `\`, punctuation, "we", "and") match the *flavor* of the paper's finding — math operators, transitional/connective words. You correctly deferred the "wait" self-reflection-token finding, since that's specific to R1-Distill-style models trained to verbalize backtracking; your 1.5B Math-Instruct model doesn't have that behavior baked in, so its absence here isn't a discrepancy, it's expected given the model choice. One thing worth noting for the writeup: your top-10 tokens only cover ~13% of all forks combined — the distribution is much flatter than a histogram like this visually suggests, worth stating explicitly to avoid overclaiming a "small set of decision tokens" narrative.

## 5. Fig. 8 — position distribution does *not* replicate, and this is worth flagging clearly

The paper reports (Appendix A) that fork positions are **roughly uniform** across the branch, which they use as a load-bearing assumption for their Theorem 1 proof. Your result is a clear U-shape: EPTree forks cluster in the first 15% of the response and spike again right at the edge of your 90% tail-mask, while random forks (as they should, by construction) stay flat.

This is a real and interesting divergence, likely coming from a stack of compounding causes rather than one bug:
- **Scale**: a 1.5B math-instruct model is probably far more uncertain at the very start of a response (before it's committed to a solution strategy) than a 14B RL-trained model would be — the paper's models had already been through RL, which tends to sharpen early-token distributions.
- **Masking rule difference** (which your own doc already flagged as a stated deviation): TreeRL masks tokens *after any mention of "answer"*, while you mask the last 10% of tokens plus after `\boxed`. These are different tail cutoffs and could produce different edge effects — your spike right before the 90% cutoff is suspicious and worth checking directly: is that a real semantic pattern (models get uncertain right before committing to a final boxed answer, e.g., second-guessing at the last step) or an artifact of tokens right at the mask boundary behaving oddly? I'd sanity check this by looking at actual text around a sample of those late forks before writing it up as a finding either way.
- **Selection band (0.1–0.9 accuracy)**: this deliberately keeps harder problems, where the model may be less settled on an approach early on, plausibly inflating early-position uncertainty relative to the paper's less-filtered set.

I would not treat this as "the replication failed" — it's a legitimate, reportable divergence, and it's actually useful information for Task 3: if uncertainty peaks at the *start* of responses, your $\mu_v+\lambda\sigma_v$ branching will also tend to fork early (your own Task 3 prediction already anticipated this), which could mean most of your search budget goes toward re-deciding solution *strategy* rather than exploring later-step errors. Worth watching for in Task 3's fork-position histogram.

## What this means for the plan

Nothing here fails your decision gate — that gate is scoped to Task 5 (oracle variance), not Task 2. Task 2's job was to confirm the harness works and characterize baseline behavior, and it did both. Two concrete adjustments I'd make before Task 3, both of which your own status doc already anticipated:

1. **Avoid comparing on saturated PassRate.** At 16 leaves your i.i.d. baseline is already 95%. Report pass@8 or pass@4-equivalent budgets alongside pass@16, and/or tighten the selection band to 0.1–0.5 accuracy as you noted, so there's real headroom left for Phase I to show separation from B0–B2.
2. **Keep paired bootstrap CIs as the default reporting unit**, not point estimates — Task 2 already shows single-seed 1-2 point differences are consistent with noise at n=243, and Phase I's expected effect sizes probably aren't going to be dramatically larger.

Otherwise, proceed to Task 3 as planned (P1-full sweep over $\lambda \in \{0, 0.5, 1, 2\}$ against the same B0/B1/B2 logs), watching specifically for: (a) whether P1 forks even earlier than B2 given the U-shape you just found, and (b) whether P1's aggregated-outcome signal produces a larger, more stable PassRate/pass@8 gap over B0 than B2 did — since that's the comparison this whole tier exists to make.