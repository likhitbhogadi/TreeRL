# GRPO baseline: code changes

GRPO (Group Relative Policy Optimization) is from DeepSeekMath (Shao et al., 2024, `deepseek_grpo.pdf`, Sec. 4.1). We add it as a third baseline next to TreeRL and ChainRL, in the same training code (see [RL_CHANGES.md](RL_CHANGES.md)). It is one new option: the default behaviour, and so every TreeRL and ChainRL run, is unchanged.

## GRPO in one paragraph

For each question, sample a group of G outputs from the current policy and score them. **Outcome supervision (Sec. 4.1.2)** gives every token of output *i* the same advantage:

  Â_i = (r_i − mean(r)) / std(r)

with the mean and std taken over that question's group. This feeds the PPO clipped objective, with a KL penalty to a frozen reference model added to the loss (β = 0.04). There is no value model. The paper uses a reward model, G = 64, learning rate 1e-6, and one policy update per sampling round.

## How it maps onto our code

GRPO's sampling is exactly our chains-only setting, `TREE="G 0 0 0"`: G independent samples through the same EPTree manager, with no forks. So GRPO and ChainRL share everything except how each response's advantage is computed:

| | ChainRL (`TREE="8 0 0 0"`) | GRPO (`TREE="8 0 0 0" ADV=grpo`) |
|---|---|---|
| Advantage of response *i* (every token) | 2 × (r_i − mean of the *other* responses): TreeRL's tree formula on trees without forks (RLOO, doubled because its global and local terms coincide) | (r_i − mean of *all* responses) / std |
| 1 correct out of 8 | correct +2.00, wrong −0.29 | correct **+2.65**, wrong −0.38 |
| 4 correct out of 8 | ±1.14 | ±1.00 |
| All right or all wrong | 0 | 0 |

GRPO puts relatively more weight on rare successes, since the std is small when a group is nearly all wrong.

The remaining parts need no change:
- **Loss:** our loss is the same clipped ratio with the same per-response length normalization. The one-sided clip bounds differ slightly (0.8 / 1.26 vs. GRPO's 0.8 / 1.2), but that never matters here. With one optimizer step per sampling round (as in the paper), the ratio is 1 (up to numerical noise) for every sample, so no clipping happens.
- **Reward:** 0/1 from the answer checker (string match, then math_verify), the same as for TreeRL/ChainRL. Most GRPO replications (e.g. DeepSeek-R1) also use rule-based rewards.

## Code changes

| File | Change |
|---|---|
| `openrlhf/trainer/ppo_utils/experience_maker.py` | New function `grpo_advantages(paths)`: reads each response's 0/1 reward (the leaf's `pass_ratio`), then writes (r − mean) / (std + 1e-6) as the value of every segment. The std is the population std over the group (the paper doesn't specify which); a group whose rewards are all equal gets 0. In `_generate_vllm_mcts`, it is called right after the tree paths are built, only if `--advantage_estimator grpo`. Everything downstream (per-token rewards, loss, training) is unchanged. |
| `train_reinforce_ray.py` | New flag `--advantage_estimator {treerl, grpo}`, default `treerl`. `_validate_args` refuses `grpo` with `--l > 0`: GRPO is defined on independent samples, not trees. |
| `scripts/treerl-qwen1.5b-1gpu.sh` | New knob `ADV` (default `treerl`), passed as `--advantage_estimator`. |
| `scripts/run_baselines.sh` | New stage: GRPO (`qwen1.5b-grpo-8`) after ChainRL, same budget, with greedy scores for all its checkpoints and 8-sample scores at steps 20 and 40. |

## Differences from the paper

| | Paper | Ours | Why |
|---|---|---|---|
| Model | DeepSeekMath-Instruct 7B | Qwen2.5-Math-1.5B-Instruct | same model as TreeRL/ChainRL |
| Group size G | 64 | 8 | matched to the other baselines (8 trained responses per question) |
| Reward | reward model | 0/1 rule-based checker | no reward model; same as the other baselines |
| KL β | 0.04, in the loss | 0 | TreeRL and ChainRL use 0, so the comparison isolates the advantage estimator. A β = 0.04 variant needs the reference model plus a KL term in the loss (~30 lines, +3 GB of GPU memory); not implemented yet. |
| Learning rate | 1e-6 | 1.5e-6 | same as the other baselines |
| Max length | 1024 | 2048 | same as the other baselines |
| Truncated responses | — | reward −1 (`--mask_repeated_samples`) | same as the other baselines |

Process-supervision GRPO (Sec. 4.1.3) needs a process reward model, and iterative GRPO (Sec. 4.1.4) retrains the reward model, so neither applies here.

## Checks

- **Advantage values.** TreeRL's own tree code was run on fake chain groups, then `grpo_advantages` (server CPU):
  - rewards [1, 0, 0, 1] → ±1.00 (ChainRL's values on the same group: ±1.33);
  - 1 correct of 8 → +2.646 / −0.378;
  - all wrong → 0.
- **Guard.** `--advantage_estimator grpo --l 1` is rejected.
- **End to end.** A 0.5B smoke run (4 questions × 4 chains, 1 step) ran through sampling, grading, the GRPO advantage, the update and the weight sync, and saved a checkpoint. It had a non-zero loss (0.19) and gradient norm (0.046).
- **TreeRL and ChainRL unaffected:** the new code runs only when `ADV=grpo`.

## How to run

```bash
# one GRPO run at the reduced budget of RL_CHANGES.md
ROLLOUT=8 NUM_TRACE=8 MAX_LEN=2048 STEPS=40 TREE="8 0 0 0" ADV=grpo TAG=qwen1.5b-grpo-8 \
    DATA=mid_submission/data/train_30k_mixed.jsonl scripts/treerl-qwen1.5b-1gpu.sh
```

`scripts/run_baselines.sh` already runs this after ChainRL; results land in `mid_submission/results/rl_eval.csv` next to the TreeRL and ChainRL rows.
