# TreeRL — EP Tree Generation Flow (Before RL Starts)

> **Scope:** Traces the code from the `.sh` launch script to the point where the
> entropy-guided EP tree is fully built and the list of training `paths` is returned.
> This is the **rollout/tree-generation phase only** — RL gradient updates happen
> *after* this.

---

## Does the code exist outside `mid_submission_*`?

**Yes.** The complete EP tree generation pipeline lives in the main source tree under
`openrlhf/`. No submission-specific folder is involved.

---

## File-by-file flow (in execution order)

### 1. `scripts/treerl-qw14b.sh`

**What it does:** The operator runs this script.  
It calls `ray job submit … -- python train_reinforce_ray.py` with all hyper-parameters,
including the key flags that enable the EP tree:

| Flag | Value | Effect |
|---|---|---|
| `--use_mcts` | (set) | activates tree generation branch |
| `--use_entropy_tree` | (set) | selects EP tree instead of vanilla MCTS |
| `--m 6` | 6 | number of initial full rollouts (M trees) |
| `--n 2` | 2 | high-entropy tokens to fork at each node (N) |
| `--l 1` | 1 | number of expansion iterations (L) |
| `--t 2` | 2 | rollouts per fork (T) |
| `--num_trace_per_sample 16` | 16 | leaf paths to keep per problem |

---

### 2. `train_reinforce_ray.py`  — `train()` (line 60)

**What it does:** The Ray job entry point.  
- Instantiates `ReinforceRayActorGroup` (actor, ref model) and vLLM engines.
- Calls `actor_model.async_fit_actor_model(...)` → dispatches to
  `ActorModelRayActorReinforce.fit.remote(...)` on every GPU actor.

---

### 3. `openrlhf/trainer/ray/launcher_reinforce.py`  
**Class `ReinforceRayActorGroup`** → `async_fit_actor_model()` (line 99)

**What it does:**  
Iterates over Ray actor handles and fires `actor.fit.remote(...)` for each one.
Returns a list of Ray futures; `train_reinforce_ray.py` waits on them with `ray.get()`.

---

### 4. `openrlhf/trainer/ray/reinforce_actor.py`  
**Class `ActorModelRayActorReinforce`** (decorated `@ray.remote`) → `fit()` (line 295)

**What it does:**  
Constructs `ActorReinforceTrainer` (a subclass of `ReinforceTrainer`) and calls
`trainer.fit(prompts_dataloader, pretrain_dataloader, args)`.

Key kwargs forwarded to the trainer (and ultimately to the experience maker):
`use_entropy_tree`, `m`, `n`, `l`, `t`, `use_pure_binary`, `use_state_value_reward`,
`use_weighted_value`, `binary_judge_url`, `extractor_url`, `reward_model_url`, etc.

---

### 5. `openrlhf/trainer/reinforce_trainer.py`  
**Class `ReinforceTrainer`** → `fit()` (line 161)

**What it does:**  
The main training loop.  
For **each prompt batch** it calls (line 207):

```python
experience = self.experience_maker.make_experience(
    rand_prompts,
    use_mcts=self.use_mcts,          # True
    use_vinevalue=self.use_vinevalue, # False in EP-tree mode
    use_sentence_level_value=False,
    **self.generate_kwargs            # contains use_entropy_tree, m, n, l, t, …
)
```

This is the call that **triggers tree generation** for the current batch.  
The returned `experience` object is appended to the replay buffer.  
The RL update (`reinforce_train()`) only happens *after* `update_timesteps` steps.

---

### 6. `openrlhf/trainer/ppo_utils/experience_maker_reinforce.py`  
**Class `RemoteExperienceMakerReinforce`** → `make_experience()` (line 140)

**What it does:**  
Checks `use_mcts=True`, `use_vinevalue=False`, so it calls:

```python
experiences = self.sample_responses_bymcts(
    prompts, num_trace_per_sample, file_name=…, **generate_kwargs
)
```

After `sample_responses_bymcts` returns, it computes KL, reward normalization,
and packages everything into an `Experience` object.  
**Tree construction itself happens entirely inside `sample_responses_bymcts`.**

---

### 7. `openrlhf/trainer/ppo_utils/experience_maker.py`  
**Class `RemoteExperienceMaker`** → `sample_responses_bymcts()` (line 2056)

**What it does:**  
Loops over each prompt in the micro-batch (one problem at a time) and calls:

```python
sequences, rewards, attention_mask, action_mask, overlong_mask, pass_ratio, pass_at_1 = \
    self._generate_vllm_mcts(batch_prompts, num_trace_per_sample, **generate_kwargs)
```

After the loop, it pads/concatenates all tensors, calls the remote reward model
(`get_remote_reward_entry_mcts`), normalises rewards, and returns a dict of tensors.

---

### 8. `openrlhf/trainer/ppo_utils/experience_maker.py`  
**Class `RemoteExperienceMaker`** → `_generate_vllm_mcts()` (line 3105)

**What it does:** The **dispatch layer** for EP vs. vanilla MCTS.

```python
use_entropy_tree = kwargs.get("use_entropy_tree", False)   # line 3124

if use_entropy_tree:
    # ──► EP tree branch
    args = {
        "temperature": …, "top_p": …,
        "m": kwargs["m"],            # initial rollouts
        "n": kwargs["n"],            # high-entropy forks per node
        "l": kwargs["l"],            # expansion iterations
        "t": kwargs["t"],            # continuations per fork
        "generate_max_len": …,
        "evaluator_urls": [judge_url],
        "extractor_urls":  [extractor_url],
        "entropy_rm_urls": [reward_model_url],
        "eos_tokens": ["<|im_end|>"],
        "num_traces": num_trace_per_sample,
        "use_pure_binary": …,
        "use_state_value_reward": …,
        "use_weighted_value": …,
        …
    }
    paths = parallel_entropy_guided_tree(
        item, llm, args, self.tokenize_fn, decode_fn, system_prompt=system_prompt
    )
else:
    # vanilla MCTS branch (parallel_mcts)
    paths = parallel_mcts(item, llm, args)
```

`paths` is a list of `num_trace_per_sample` root-to-leaf paths, each annotated
with token IDs, rewards, and advantage values.

---

### 9. `openrlhf/trainer/ppo_utils/entropy_guided_tree_search.py`  
**`parallel_entropy_guided_tree()`** (line 70)

**What it does:** Thin wrapper — creates `EntropyGuidedChainLocalManager` and
calls `manager.process_single_item(item, args)`.

```python
manager = EntropyGuidedChainLocalManager(
    args=args, llm=llm,
    encode_fn=tokenize_fn, decode_fn=decode_fn,
    evaluator_urls=args['evaluator_urls'],
    extractor_urls=args['extractor_urls'],
    eos_tokens_set=args['eos_tokens'],
)
result = manager.process_single_item(item, args)
paths  = result["paths"]
```

---

### 10. `openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py`  
**Class `EntropyGuidedChainLocalManager`**

This is where the **EP tree is actually built**. Two methods matter:

#### `process_single_item()` (line 447)
Calls `self.entropy_guided_chain(problem, answer, args=args)` and wraps
the result into a dict.

#### `entropy_guided_chain()` (line 154)  ← **core algorithm**

```
Step A — Initial rollouts (M full responses)
  • Tokenize the prompt → init_prompt_ids
  • Call query_local_vllm_ids_with_logprobs(init_prompt_ids × M)
  • Each result becomes a TreeNode (root of one of the M trees)

Step B — Expansion loop (L iterations)
  For each iteration:
    1. For every node in every tree:
         pick the top-N tokens by highest entropy
         (= highest negative log-probability)
    2. Build (expansion_tasks × T) prompts:
         prefix = prompt + tokens up to the high-entropy split point
    3. Batch-infer continuations with vLLM
    4. Each result becomes a child TreeNode attached to its parent

Step C — Parallel evaluation
  • self.evaluate_trees() scores every leaf node:
      - binary_judge (LLM evaluator) → binary_score (0/1)
      - reward_model (RM) → sigmoid(RM_score)
      - final_score = binary + 0.5 × sigmoid  (or use_pure_binary / use_pure_RM)

Step D — Path assembly
  • build_into_tree_format(self.tree_lists, …)
      → connects M tree-lists into a unified root/child structure,
        assigns value/advantage to each node            [tree_node.py]
  • gather_paths(root, selected_terminals, pass_k=num_traces, …)
      → traverses from root to each selected leaf,
        collects (token_ids, reward, advantage, overlong_mask) per path
                                                         [parallel_mcts.py]
```

---

### 11. `openrlhf/trainer/ppo_utils/tree_node.py`  
**`build_into_tree_format()`** (line 208)

Converts the flat `tree_lists` (list-of-lists of `TreeNode` objects) into a
proper tree with `root` / `child_nodes` pointers. Also assigns
value / advantage to nodes according to `use_weighted_value`,
`average_one_generation`, `inner_repetition_penalty`, etc.  
Returns `(root, selected_terminals)`.

---

### 12. `openrlhf/trainer/ppo_utils/parallel_mcts.py`  
**`gather_paths()`** (line 1622)

Walks from root to each selected terminal leaf, collecting a path dict per leaf:
- `sequences` (token IDs of the full response)
- `reward` (step-level or sequence-level)
- `advantage`
- `overlong_mask`

Returns `List[List[Dict]]` — the `paths` object that bubbles all the way back up to
`sample_responses_bymcts`, which then converts it into tensors for RL.

---

## Summary diagram

```
scripts/treerl-qw14b.sh
  └─ python train_reinforce_ray.py
       └─ train()
            └─ ReinforceRayActorGroup.async_fit_actor_model()     [launcher_reinforce.py]
                 └─ ActorModelRayActorReinforce.fit()             [reinforce_actor.py]
                      └─ ReinforceTrainer.fit()                   [reinforce_trainer.py]
                           └─ [per-batch] experience_maker.make_experience()
                                └─ RemoteExperienceMakerReinforce.make_experience()
                                     └─ [use_mcts=True] sample_responses_bymcts()
                                          └─ _generate_vllm_mcts()
                                               └─ [use_entropy_tree=True]
                                                  parallel_entropy_guided_tree()   [entropy_guided_tree_search.py]
                                                    └─ EntropyGuidedChainLocalManager.process_single_item()
                                                         └─ entropy_guided_chain()  [entropy_chain_local_manager.py]
                                                              ├─ A. M initial vLLM rollouts  → M TreeNode roots
                                                              ├─ B. L expansion iterations   → child TreeNodes (high-entropy forks)
                                                              ├─ C. evaluate_trees()         → binary + RM scores on leaves
                                                              ├─ D. build_into_tree_format() [tree_node.py]
                                                              └─ E. gather_paths()           [parallel_mcts.py]
                                                                      └─ returns List[paths]
                                                              ← paths back to sample_responses_bymcts
                                          ← tensors (sequences, rewards, action_mask, …)
                                ← Experience object (sequences, advantages, action_log_probs, …)
                           ← appended to replay_buffer
                      ← reinforce_train() called (RL update begins here ↑)
```

---

## Key files — quick reference

| File | Role |
|---|---|
| `scripts/treerl-qw14b.sh` | Launch script; sets all hyper-params |
| `train_reinforce_ray.py` | Ray job entry point |
| `openrlhf/trainer/ray/launcher_reinforce.py` | `ReinforceRayActorGroup` — distributes `fit` across GPU actors |
| `openrlhf/trainer/ray/reinforce_actor.py` | `ActorModelRayActorReinforce` — per-GPU actor; wires up trainer |
| `openrlhf/trainer/reinforce_trainer.py` | `ReinforceTrainer.fit()` — the main episode loop |
| `openrlhf/trainer/ppo_utils/experience_maker_reinforce.py` | `RemoteExperienceMakerReinforce.make_experience()` — routes to MCTS or vanilla |
| `openrlhf/trainer/ppo_utils/experience_maker.py` | `sample_responses_bymcts()` + `_generate_vllm_mcts()` — dispatch to EP tree |
| `openrlhf/trainer/ppo_utils/entropy_guided_tree_search.py` | `parallel_entropy_guided_tree()` — thin wrapper |
| `openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py` | `EntropyGuidedChainLocalManager` — **core EP tree algorithm** |
| `openrlhf/trainer/ppo_utils/tree_node.py` | `TreeNode` class + `build_into_tree_format()` |
| `openrlhf/trainer/ppo_utils/parallel_mcts.py` | `gather_paths()` — extracts root-to-leaf path tensors |
