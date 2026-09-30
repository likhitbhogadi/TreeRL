# RL part: TreeRL training on one GPU (code changes)

We reuse TreeRL's released RL code (`train_reinforce_ray.py` + `openrlhf/`, a fork of OpenRLHF): EPTree rollouts, tree-derived per-token advantages, REINFORCE loss, DeepSpeed, Ray, vLLM. Only what cannot run on our machine is changed:

- **Hardware:** one 46 GB L40S, shared with other users (the paper used 16+ GPUs).
- **Library versions:** vLLM 0.30, transformers 5.17, DeepSpeed 0.19 (the released code targets vLLM ≈0.4–0.6 and transformers 4.x).

The algorithm itself is untouched: `tree_node.py`, `parallel_mcts.gather_paths`, `entropy_chain_local_manager.py` (apart from the Task 2 `random_fork` flag), `models/loss.py` and `replay_buffer.py` are unchanged. Every edit has a comment at the changed line, and `git diff` shows the full diff.

Status: runs end to end. It was checked on Qwen2.5-0.5B-Instruct, with 3 training steps across two smoke runs, checkpoint save and reload, and the weight-sync check below. The 1.5B run is waiting for GPU memory; see [VRAM](#vram-does-15b-fit-in-40-gb).

## How to run

On pkgpu2, the code is in `~/likhit/tree_based_rl/TreeRL_rl/` (a copy of this repo; the uv venv `~/likhit/.venv` now also has `deepspeed datasets accelerate`).

```bash
# weight-sync self-check (~2 min, ~7 GB of GPU memory)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/check_vllm_weight_sync.py

# TreeRL, Qwen2.5-Math-1.5B-Instruct, EPTree (6,2,1,2), 16 prompts x 16 leaves per step
GPU=1 STEPS=100 nohup scripts/treerl-qwen1.5b-1gpu.sh > /tmp/treerl_rl.log 2>&1 &
```

- **Outputs:** checkpoints go to `ckpt/<TAG>/_actor_global_step<N>` every `SAVE_STEPS` (20). One line of metrics per step goes to `ckpt/<TAG>/train_log.jsonl`: reward, `pass_at_1` (mean leaf accuracy), `pass_rate` (share of trees with a correct leaf), response length, generate/rollout time, grad norm.
- **Knobs** (environment variables): `GPU`, `STEPS`, `ROLLOUT` (prompts per step), `NUM_TRACE` (leaves per tree used for training), `VLLM_MEM` (share of the GPU vLLM takes while generating), `INFER_BS` (batch of the logprob pass), `MODEL`, `TAG`, `SAVE_DIR`, `SAVE_STEPS`.

The script uses the flags of the paper's `scripts/treerl-qw14b.sh`, with these differences:

| Setting | Paper | Ours | Why |
|---|---|---|---|
| GPUs | 16 actor + 16 vLLM + 8 ref | 1 (everything colocated) | hardware |
| Model | Qwen2.5-14B (SFT) | Qwen2.5-Math-1.5B-Instruct | VRAM; same model as Task 2 |
| Temperature | 1.2 | 1.0 | Task 2: at 1.2 this model degenerates in ~13% of samples |
| Max response | 8192 | 3072 (prompt 1024) | the model's context is 4096 |
| ZeRO | 3 | 2 (+ CPU Adam, as in the paper) | one GPU; ZeRO-3 only helps across GPUs |
| Judge / extractor / RM servers | LLM judge over HTTP | local `\boxed{}` grading (string match, then math_verify) | no servers |
| Run length | 2 epochs of 30k prompts | `STEPS` × 16 prompts | time (see below) |

Unchanged from the paper: EPTree (6,2,1,2), 16 of the 30 leaves per tree, `--process_supervision --use_state_value_reward --use_pure_binary --use_weighted_value sqrt --mask_repeated_samples`, KL 0, lr 1.5e-6, weight decay 0.1, train batch 256 = one optimizer step per rollout, and data `datasets/train/train_30k.jsonl`.

## What changed, file by file

### 1. Running on one GPU with vLLM 0.30 (the main change)

In the released design, actor, reference model and each vLLM engine are Ray actors on separate GPUs. After each update the actor broadcasts its weights to vLLM over NCCL, through a subclass of `vllm.worker.worker.Worker`. On our setup both parts break:
- **NCCL:** NCCL refuses two ranks on one GPU.
- **vLLM:** `vllm.worker.worker.Worker` no longer exists in vLLM's V1 engine.

We follow upstream OpenRLHF's later "colocate" design instead.

| File | Change |
|---|---|
| `train_reinforce_ray.py` | Actor and vLLM are Ray actors with a GPU fraction (0.3), so all of them land on the one visible GPU. New `--vllm_gpu_memory_utilization`. `max_model_len` = prompt + generate length (the old hard-coded 23856 is above the model's 4096). `--enable_prefix_caching` is now passed through (it was parsed but ignored); it helps EPTree, whose forks share prefixes. No reference model when `--init_kl_coef 0` (see §3). |
| `trainer/ray/vllm_engine.py` | `LLMRayActor` rewritten for vLLM V1. The engine runs in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) with a worker extension and sleep mode. `generate()` accepts the old `prompt_token_ids=` call (removed in vLLM ≥0.10) and converts it to `TokensPrompt`. `sleep()` / `wake_up()` use level 1: weights to CPU, KV cache freed. `update_weights_cuda_ipc()` loads the new weights, then resets the prefix cache (cached KV came from the old weights). vLLM's FlashInfer sampler is disabled, as in Task 2's `gen.py` (it JIT-compiles with the system nvcc 12.0, too old for torch cu129). One colocated TP=1 engine only. |
| `trainer/ray/vllm_worker_wrap.py` | Old `Worker` subclass replaced by a `worker_extension_cls` with one method. It rebuilds the actor's parameters from CUDA IPC handles and calls `model.load_weights`. There is no copy and no NCCL. |
| `trainer/ray/reinforce_actor.py` | NCCL process-group setup removed. `_broadcast_to_vllm` now sends `reduce_tensor(param)` IPC handles of all parameters in one call, after waking the engine. ZeRO-1/2 only (under ZeRO-3 the parameters are sharded). |
| `trainer/ray/launcher_reinforce.py` | Accepts a missing reference-model group. |

**Memory timeline per step:**
1. vLLM builds the trees (awake).
2. vLLM sleeps.
3. The actor computes old logprobs, then trains (256 micro-batches, one optimizer step).
4. Weight sync wakes vLLM and loads the new weights.

**Verified by `scripts/check_vllm_weight_sync.py`.** It perturbs an HF copy of the model, pushes it through sleep → wake → IPC update, and compares vLLM's token logprobs with HF's. The max gap is 0.04 with identical weights, 0.2–0.5 while vLLM is stale, and 0.007–0.05 after sync, i.e. back to the bf16 noise level.

### 2. Speed: build the rollout batch's trees concurrently

In `trainer/ppo_utils/experience_maker.py`, `sample_responses_bymcts` built one EPTree at a time. Each tree issues a few vLLM calls of only 6–24 sequences, so vLLM sat mostly idle. The paper hid this behind 16 vLLM engines and 16 actor ranks, each handling one prompt.

Now all prompts of the rollout batch build their trees in parallel threads through `BatchingLLM`, a ~40-line class in the same file. Each thread's `generate()` waits until every unfinished tree has submitted, then all prompts go to vLLM as one batch. EPTree's rounds (M chains, then each fork round) line up across trees, so each round of all 16 trees becomes one vLLM call. The released per-tree code (`_generate_vllm_mcts`, the manager) is unchanged apart from accepting this `llm` object.

To get the old behaviour back, set `--micro_rollout_batch_size 1`.

A second speed-up is in `trainer/reinforce_trainer.py`. The trainer computed a logging-only gradient norm after *every micro-batch*, looping `safe_get_full_grad(p).item()` over all ~300 parameters. That is ~300 GPU syncs and CPU-offload reads per sample, of a half-accumulated gradient. It now reads DeepSpeed's own global norm, which DeepSpeed computes for clipping at each optimizer step. Measured on 0.5B: training went from 3.7 s to 1.5 s per sample.

### 3. Other changes

| File | Change | Why |
|---|---|---|
| `experience_maker.py`, `train_reinforce_ray.py` | With `--init_kl_coef 0` there is no reference model; its logprobs are replaced by the actor's (KL = 0). | Paper setting. The reference model only feeds the KL term, so this saves ~3 GB and one forward pass per sample. KL > 0 still works (reference model colocated). |
| `experience_maker.py` | `pass_rate` / `pass_at_1` logged per sample. | The released code logged the *last* tree's value for the whole batch (harmless there with 1 prompt per call). |
| `evaluation.py` | Grading without judge servers: normalized string match, then math_verify in a spawned process pool (10 s timeout, pool replaced on a hang). | On our Task 2 trees the string match alone misses **14.4%** of the answers math_verify accepts (8,532 of 59,179, from `results/csv/nodes.csv`), which would be reward noise. Grading runs in threads, where math_verify's own signal-based timeout can't work and SymPy can hang forever; a pool worker can be killed. |
| `trainer/reinforce_trainer.py` | Metrics also written to `<save_path>/train_log.jsonl`. | wandb is not installed; this gives one JSON line per step. |

### 4. Bugs in the released code that our setup exposed

| File | Bug | Fix |
|---|---|---|
| `experience_maker.py` `_tokenize_fn_llama` | Appended DeepSeek-R1's `"<｜Assistant｜>"` marker to *every* chat model's prompt, which is garbage tokens for Qwen. | `apply_chat_template(..., add_generation_prompt=True)`. The Qwen prompt is now byte-identical to Task 2's (`<|im_start|>assistant\n`, with Qwen2.5-Math's default "reason step by step … `\boxed{}`" system prompt). |
| `experience_maker.py` `_generate_vllm_mcts` | Passed `["<|im_end|>"]` (a string) as vLLM's `stop_token_ids`. | Token ids of `<|im_end|>` and `<|endoftext|>`. |
| `evaluation.py` `check_result` | `--binary_judge_url` unset gives `[None]`, which counted as "servers present", so grading would query URL `None` forever. | Treat `[None]` as no servers. |
| `utils/remote_reward.py` | `torch.tensor(results)` on per-token rewards of different trees crashes: each tree is padded to its own response length. It never triggered with one prompt per call. | Right-pad with 0, like the action masks. |
| `utils/deepspeed.py` `save_model` | Tied-embedding models (all Qwen2.5 ≤ 3B) failed the state-dict size assert before reaching the tied-embedding special case written for them. | Count the tied `lm_head` in the assert. |
| `models/actor.py`, `models/model.py`, `reinforce_trainer.py` | `transformers.deepspeed` and `tokenizer.additional_special_tokens` no longer exist in transformers 5. | `transformers.integrations`; `len(tokenizer)`. |

Not fixed (harmless for us): `entropy_guided_tree_search.parallel_entropy_guided_tree` drops its `system_prompt` argument, so trees are always built without `--system_prompt`. Don't pass `--system_prompt`; Qwen2.5-Math's template already adds the right one.

## VRAM: does 1.5B fit in 40 GB?

**Yes.** Qwen2.5-Math-1.5B has 1.54B parameters.

| | GPU memory |
|---|---|
| Actor weights, bf16 | 3.1 GB |
| Actor gradients, bf16 (ZeRO-2) | 3.1 GB |
| Adam m, v + fp32 master weights (12 B/param) | 0 on GPU with `--adam_offload` (18.5 GB without it) |
| Training activations: one 4k-token sequence, gradient checkpointing, 152k-vocab logits | ~5–8 GB peak |
| vLLM while generating: weights 3.1 + KV cache + CUDA graphs | `VLLM_MEM` × 46 GB (0.35 → 16 GB); freed outside generation |
| Reference model | 0 (KL = 0) |

- **Peak while generating:** 6.2 + 16 ≈ 22 GB, plus CUDA contexts, so about 25 GB.
- **Peak while training:** ≈ 15 GB.
- **Without CPU Adam:** about 43 GB. It would not fit next to vLLM, which is why `--adam_offload` stays on (the paper uses it too).
- **Inference vs. RL:** EPTree inference alone needed only the 3 GB of weights plus KV cache. RL adds the gradients, optimizer state and training activations.

A 0.5B or 1B model is not needed. Qwen2.5-Math has no 0.5B/1B version anyway; the 0.5B above is the general Qwen2.5-0.5B-Instruct, used only for the smoke test.

The smoke tests had to fit into the ~11 GB other users left free, which is why they use the 0.5B model, `VLLM_MEM=0.12` and `INFER_BS=1`.

## Time

Measured on 0.5B, on a GPU another job was using at 100%:
- A rollout of 4 trees took 1–2 min.
- Training took 1.5 s per sample.

Estimate for 1.5B, 16 prompts × 16 leaves = 256 samples per step:
- **Per step:** rollout ~2–4 min plus training ~4–6 min, so **~8–10 min on a free GPU**, and more when shared.
- **100 steps** (1,600 problems): **~14–17 h**.

The paper's schedule (2 epochs over 30k problems) is out of reach on one GPU, so runs use a fixed `STEPS` budget. Everything we compare should use the same budget.

## Known limitations / next steps

- **No evaluation during training** (the released code has a TODO there). Evaluate saved checkpoints on MATH500 / AMC / Omni-MATH-500 with vLLM, e.g. by reusing `tree_alloc/gen.py` and `verify.py`.
- **Single GPU only.** Multi-GPU would need the NCCL weight sync ported to vLLM V1.
- **ChainRL baseline:** same script with `--m 16 --n 0 --l 0 --t 0`, i.e. i.i.d. chains through the same code path. Not run yet.
- **Metric naming:** `response_overlong_ratio` in the log is really the share of responses that ended properly (the code's `overlong_mask` is 1 for a finished response).
- **Our method:** Phase I / Phase II allocation plugs in where EPTree picks fork points (`entropy_chain_local_manager.py`). Everything downstream of the tree (advantages, loss, training) is reused as is.
