# TreeRL — RL Update + Evaluation Flow (After EP Tree is Built)

> **Scope:** Starts exactly where the previous trace ended — after `sample_responses_bymcts()`
> returns an `Experience` object — and follows the code all the way through the gradient
> update, weight broadcast back to vLLM, checkpoint saving, and then the separate
> downstream evaluation pipeline.
>
> **Answer:** Yes, all of this code exists outside the `mid_submission_*` folders.

---

## Part A — RL Update (weight update from TreeRL paths)

### Flow diagram

```
reinforce_trainer.py  fit()
  └─ [every update_timesteps steps]
       ├─ replay_buffer.append(experience)                  [replay_buffer.py]
       ├─ reinforce_train(global_step)                      [reinforce_actor.py overrides this]
       │    ├─ experience_maker.flush()
       │    ├─ super().reinforce_train()                    [reinforce_trainer.py]
       │    │    └─ [for each batch from replay buffer]
       │    │         ├─ training_step(experience)
       │    │         │    └─ training_step_actor(experience)
       │    │         │         ├─ actor forward pass  → action_log_probs
       │    │         │         ├─ ReinforcePolicyLoss.forward()    [models/loss.py]
       │    │         │         ├─ strategy.backward(loss)          [DeepSpeed ZeRO step]
       │    │         │         ├─ strategy.optimizer_step()        [Adam + cosine LR scheduler]
       │    │         │         └─ (optional EMA update)
       │    │         └─ strategy.all_reduce(status)
       │    └─ _broadcast_to_vllm()                        [reinforce_actor.py]
       │         └─ NCCL broadcast updated weights → all vLLM engines
       ├─ kl_ctl.update()
       └─ save_logs_and_checkpoints()
            ├─ wandb.log(metrics)
            └─ strategy.save_model(actor, tokenizer, ckpt_path/_actor_global_stepN)
```

---

### File-by-file (RL update only)

#### 1. `openrlhf/trainer/ppo_utils/replay_buffer.py`  
**`NaiveReplayBuffer`** — `append()` / `collate_fn()` / `normalize()`

After `make_experience()` returns, `ReinforceTrainer.fit()` calls:
```python
self.replay_buffer.append(experience)
```
- `split_experience_batch()` unpacks the batched `Experience` into individual `BufferItem`s.
- `remove_padding_in_sequences()` strips left/right padding per item.
- Items accumulate until `update_timesteps` is reached.

When the buffer is full, `reinforce_train()` wraps it in a `DataLoader` using
`collate_fn` → `make_experience_batch()` which re-pads and returns a new batched
`Experience` for each mini-batch.

---

#### 2. `openrlhf/trainer/reinforce_trainer.py`  
**`ReinforceTrainer.reinforce_train()`** (line 250)

Called once per `update_timesteps` steps.  
Creates a DataLoader from the buffer, loops over epochs and mini-batches, calls
`training_step()` → `training_step_actor()` for each mini-batch.

**`training_step_actor()`** (line 311) — the actual gradient update:

```python
# 1. Forward pass through current actor
action_log_probs, output = self.actor(
    experience.sequences, num_actions,
    attention_mask=experience.attention_mask,
    return_output=True
)

# 2. REINFORCE policy loss
actor_loss = self.actor_loss_fn(
    action_log_probs,        # π_θ(a|s) from current policy
    experience.action_log_probs,  # π_old(a|s) from tree rollout (frozen)
    experience.advantages,   # reward signal from TreeRL tree paths
    action_mask=experience.action_mask,
    kl=experience.kl,
    kl_coef=self.strategy.args.init_kl_coef,
)

# 3. Backward + optimizer step (DeepSpeed handles ZeRO sharding)
self.strategy.backward(loss, self.actor, self.actor_optim)
self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler)

# 4. (optional) EMA weight update
if self.ema_model:
    self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")
```

After all mini-batches: `strategy.all_reduce(status)` gathers metrics across ranks.

---

#### 3. `openrlhf/models/loss.py`  
**`ReinforcePolicyLoss`** (line 80)

The loss function used by the TreeRL REINFORCE update:

```python
ratio = (log_probs - old_log_probs).exp()   # importance weight π_θ / π_old
surr1 = ratio * rewards                      # unclipped objective
surr2 = ratio.clamp(1 - eps, 1 + eps*1.3) * rewards  # clipped objective
loss  = -torch.min(surr1, surr2)             # PPO-style clipping
loss  = masked_mean(loss, action_mask).mean()
```

Here `rewards` = `experience.advantages` = the token-level advantage values
produced by `gather_paths()` in the tree generation phase.

> **Note:** `PolicyLoss` (line 28) is the standard PPO loss used only when
> `use_vinevalue=True`. For the EP tree mode, it's always `ReinforcePolicyLoss`.

---

#### 4. `openrlhf/trainer/ray/reinforce_actor.py`  
**`ActorReinforceTrainer.reinforce_train()`** (line 102) — overrides the base class

After the gradient step finishes, this method does:

```python
# 1. Flush experience maker (no-op in practice)
self.experience_maker.flush()

# 2. Run the actual training (base class)
status = super().reinforce_train(global_step)

# 3. Broadcast updated weights back to ALL vLLM engines
if self.vllm_engines is not None:
    self._broadcast_to_vllm()

torch.distributed.barrier()
```

**`_broadcast_to_vllm()`** (line 129):
- Iterates over every named parameter of the actor model.
- For ZeRO-1/2: rank 0 broadcasts directly via NCCL (`torch.distributed.broadcast`).
- For ZeRO-3: rank 0 first `AllGather`s the sharded param, then broadcasts.
- Each vLLM engine calls `engine.update_weight.remote(name, dtype, shape)` to receive the new weights.

This closes the loop: after every RL update, the vLLM engines that were used for
tree generation get the updated policy weights so the next iteration uses the new model.

---

#### 5. `openrlhf/trainer/reinforce_trainer.py`  
**`save_logs_and_checkpoints()`** (line 440)

After `reinforce_train()` returns:
```python
# Log to W&B
if global_step % args.logging_steps == 0:
    wandb.log({"train/policy_loss": …, "train/reward": …, …})

# Save model checkpoint (HuggingFace format) at save_steps interval
if global_step % args.save_steps == 0:
    tag = f"global_step{global_step}"
    self.strategy.save_model(
        self.actor.model, self.tokenizer,
        os.path.join(args.ckpt_path, f"_actor_{tag}")
    )
```

Checkpoints are saved as HuggingFace `.safetensors` files at:
```
$SAVE_DIR/_actor_global_step{N}/
```

---

## Part B — Evaluation / Metrics Calculation

The evaluation pipeline is **separate** from training — you run it after training
finishes (or at any checkpoint) using either `batch_inference.py` or the dedicated
evaluation scripts under `evaluation/`.

> **Important note:** The `evaluation/` scripts (`ceval`, `cmmlu`, `gpt4`) are
> generic LLM evaluation harnesses that predate this project; they are not math-RL
> specific. The TreeRL-specific "online" eval that happens *during* training is just
> the `pass_rate` / `pass_at_1` logged to W&B from the tree generation phase.
> No eval loop is wired into `reinforce_trainer.py` (the `eval_steps` branch is a
> `pass` stub at line 456–458).

---

### Evaluation flow diagram

```
[Step 1: Generate responses]
  batch_inference.py  batch_generate_vllm()   ← run with --eval_task generate_vllm

[Step 2a: Math/benchmark evaluation]
  evaluation/ceval/run_ceval.sh
    └─ ceval.py  main()
         └─ Llama_Evaluator.eval_subject()    [llama_evaluator.py]
              └─ model.generate() → constrained decoding → accuracy per subject

  evaluation/cmmlu/run_cmmlu.sh
    └─ eval.py  main()
         └─ llama2_evaluator.py (same pattern as ceval)

[Step 2b: GPT-4 win-rate evaluation]
  alpaca_eval  (external tool, called from evaluation/gpt4/README.md)
    └─ compares model outputs vs. reference using GPT-4 as judge
```

---

### File-by-file (evaluation)

#### 6. `batch_inference.py`  — `batch_generate_vllm()` (line 17)

**What to run:**
```bash
python batch_inference.py \
  --eval_task generate_vllm \
  --pretrain $SAVE_DIR/_actor_global_step{N} \
  --dataset datasets/eval/your_eval.jsonl \
  --output_path outputs/eval_results.jsonl \
  --max_new_tokens 8192 \
  --tp_size 8
```

**What it does:**
1. Loads tokenizer + vLLM `LLM` from the saved checkpoint.
2. Creates `SamplingParams` (top_p, temperature, max_tokens).
3. Loads the eval dataset via `blending_datasets()`.
4. Calls `llm.generate(prompts, sampling_params)` → saves `{input, output}` pairs to a `.jsonl` file.

For reward model scoring, use `--eval_task rm` which calls `batch_rm_inference()`.

---

#### 7. `evaluation/ceval/run_ceval.sh` + `ceval.py` + `llama_evaluator.py`

**What to run:**
```bash
cd evaluation/ceval
bash run_ceval.sh    # edit the model path inside first
```

**`ceval.py` → `main()`:**
- Iterates over all 52 CEval subjects (loaded from `ceval_data/val/`).
- For each subject, calls `Llama_Evaluator.eval_subject()`.
- Aggregates per-subject accuracy, groups by domain (STEM / Social / Humanities / Other).
- Writes `summary.json` and `submission.json`.

**`llama_evaluator.py` → `Llama_Evaluator.eval_subject()`:**
- Loads the model using HuggingFace `LlamaForCausalLM` (swap for your model class).
- Runs **constrained decoding**: generates just 1 token, reads the logits for A/B/C/D
  tokens, picks the highest → no open-ended generation needed for MCQ.
- Returns `correct_ratio` (%) per subject.

> **Key metric:** Overall accuracy across all 52 subjects.

---

#### 8. `evaluation/cmmlu/run_cmmlu.sh` + `eval.py` + `llama2_evaluator.py`

Identical structure to CEval but uses the CMMLU benchmark (67 Chinese language subjects).

```bash
cd evaluation/cmmlu
bash run_cmmlu.sh
```

`eval.py` → `llama2_evaluator.py` → `Llama2_Evaluator.eval_subject()`.

> **Key metric:** Per-category and overall CMMLU accuracy.

---

#### 9. `evaluation/gpt4/` — GPT-4 Win-Rate

**What to run (two steps):**

```bash
# Step 1: generate responses with trained model
deepspeed batch_inference.py \
  --eval_task generate \
  --pretrain $CKPT \
  --dataset evaluation/gpt4/benchmark.jsonl \
  --output_path outputs/model_outputs.jsonl

# Step 2: GPT-4 judge evaluation
alpaca_eval --model_outputs outputs/model_outputs.jsonl \
            --annotators_config alpaca_eval_gpt4 \
            --reference_outputs sft.json
```

The benchmark data lives at `evaluation/gpt4/benchmark.jsonl` (160 prompts from LMSys).  
Metrics: **Win / Lose / Tie** rate vs. reference model (as judged by GPT-4).

---

## Complete end-to-end summary (both parts)

```
┌─────────────────────────────── TRAINING LOOP ────────────────────────────────┐
│                                                                               │
│  [Tree generation — previous trace]                                           │
│       ↓  returns Experience(sequences, advantages, action_log_probs, …)       │
│                                                                               │
│  replay_buffer.append(experience)              [replay_buffer.py]             │
│       ↓  (accumulate update_timesteps batches)                                │
│                                                                               │
│  reinforce_train(global_step)                  [reinforce_actor.py]           │
│    ├─ super().reinforce_train()                [reinforce_trainer.py]         │
│    │    └─ training_step_actor()                                              │
│    │         ├─ actor.forward()  → action_log_probs                           │
│    │         ├─ ReinforcePolicyLoss(log_probs, old_log_probs, advantages)     │
│    │         │       = -min(ratio*adv, clip(ratio)*adv)    [models/loss.py]   │
│    │         ├─ strategy.backward(loss)  [DeepSpeed ZeRO-3 backward]          │
│    │         └─ strategy.optimizer_step()  [Adam + cosine LR]                 │
│    │                                                                           │
│    └─ _broadcast_to_vllm()                    [reinforce_actor.py]            │
│         └─ NCCL broadcast new weights → all vLLM engines                     │
│                                                                               │
│  save_logs_and_checkpoints()                   [reinforce_trainer.py]         │
│    ├─ wandb.log(policy_loss, reward, kl, pass_rate, …)                        │
│    └─ strategy.save_model() → $CKPT/_actor_global_stepN/                     │
│                                                                               │
│       ↓  (next episode — tree generation uses updated vLLM weights)           │
└───────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────── EVALUATION ───────────────────────────────────┐
│                                                                               │
│  [Offline — run after training or at any checkpoint]                          │
│                                                                               │
│  batch_inference.py  batch_generate_vllm()                                   │
│    └─ vLLM generate from checkpoint → outputs.jsonl                          │
│                                                                               │
│  evaluation/ceval/ceval.py  main()                                            │
│    └─ Llama_Evaluator.eval_subject()          [llama_evaluator.py]            │
│         └─ constrained decoding (1 token, A/B/C/D) → accuracy per subject    │
│              → summary.json  (STEM / Social / Humanities / Overall)          │
│                                                                               │
│  evaluation/cmmlu/eval.py  main()                                             │
│    └─ Llama2_Evaluator.eval_subject()         [llama2_evaluator.py]           │
│         └─ same approach → per-category CMMLU accuracy                       │
│                                                                               │
│  alpaca_eval  (external)                      [evaluation/gpt4/]              │
│    └─ GPT-4 judges model outputs vs. reference → Win/Lose/Tie rate           │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Key files — quick reference

| File | Role |
|---|---|
| `openrlhf/trainer/ppo_utils/replay_buffer.py` | Buffer that accumulates Experience objects; splits/re-pads for mini-batches |
| `openrlhf/trainer/reinforce_trainer.py` | `reinforce_train()` + `training_step_actor()` — the gradient update loop |
| `openrlhf/models/loss.py` | `ReinforcePolicyLoss` — the clipped REINFORCE objective |
| `openrlhf/trainer/ray/reinforce_actor.py` | `ActorReinforceTrainer.reinforce_train()` — wraps base class; calls `_broadcast_to_vllm()` |
| `openrlhf/trainer/reinforce_trainer.py` | `save_logs_and_checkpoints()` — W&B logging + HF checkpoint saving |
| `batch_inference.py` | Offline response generation from a saved checkpoint using vLLM |
| `evaluation/ceval/ceval.py` | CEval benchmark orchestration (52 Chinese knowledge subjects) |
| `evaluation/ceval/llama_evaluator.py` | Per-subject constrained-decoding evaluator |
| `evaluation/cmmlu/eval.py` | CMMLU benchmark orchestration (67 subjects) |
| `evaluation/cmmlu/llama2_evaluator.py` | Per-subject evaluator for CMMLU |
| `evaluation/gpt4/benchmark.jsonl` | 160-prompt benchmark for GPT-4 win-rate evaluation |
