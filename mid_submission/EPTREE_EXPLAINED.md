# How TreeRL builds its trees (EPTree), explained

This covers two functions and the pipeline around them:

1. [`entropy_guided_chain()`](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L154) builds the tree.
2. [`query_local_vllm_ids_with_logprobs()`](../openrlhf/trainer/ppo_utils/evaluation.py#L775) asks vLLM for text plus a log-probability for every token.
3. The pipeline: who calls the tree builder, and what happens to the tree afterwards (in RL training, and in our own `run_treerl.py`).

Every file reference is a link you can click.

---

## 0. The one-paragraph version

For each math problem, the model writes **M** complete answers. For each answer we look at every token and ask *"how surprised was the model by the token it picked?"* (`-logprob`). The **N** most surprising positions in each tree become **fork points**. From each fork point we cut the answer just *before* that token and let the model write **T** new endings. We repeat this **L** times. At the end every branch is graded (right or wrong). Those grades are pushed up the tree so that every *segment* of text gets its own reward, and that per-token reward trains the model.

```
problem ──► root answer 1: "Let x = 2 . So we get 7"             ✓
                                     │
                           fork here (model was unsure at "So")
                                     └──► "Thus we need 3"        ✗
        ──► root answer 2: ...
```

---

## 1. Where the "entropy" comes from (short answer)

**There are no logits in this code, and no real entropy either.**

| Step | File | What happens |
|---|---|---|
| 1 | [vllm_engine.py:45-48](../openrlhf/trainer/ray/vllm_engine.py#L45-L48) | A normal `vllm.LLM` runs the model. The logits exist only inside vLLM. |
| 2 | [evaluation.py:787](../openrlhf/trainer/ppo_utils/evaluation.py#L787) | `SamplingParams(logprobs=True)` asks vLLM to also return the **log-probability of each sampled token**. |
| 3 | [evaluation.py:825-846](../openrlhf/trainer/ppo_utils/evaluation.py#L825-L846) | For each generated position, the code keeps the token ID and its logprob. |
| 4 | [entropy_chain_local_manager.py:210-221](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L210-L221) | These go into `TreeNode(log_prob_list=...)`. |
| 5 | [tree_node.py:182-204](../openrlhf/trainer/ppo_utils/tree_node.py#L182-L204) | `get_max_entropy_tokens()` computes `entropy = -log_prob` and ranks positions. |

So the "entropy" is really **surprisal** (also called token NLL): how unlikely the one token that was *actually chosen* was. It is not the entropy of the whole next-token distribution (`-Σ p·log p` over the vocabulary).

- Probability 0.9 gives surprisal 0.105 (the model was sure, so it's a bad fork point).
- Probability 0.02 gives surprisal 3.9 (the model was unsure, so it's a good fork point).

If you ever want *true* entropy, you have two options:
- Request top-k logprobs (for example `logprobs=20`) and approximate the sum over those k tokens.
- Compute it yourself from the model's logits, outside vLLM.

---

## 2. `query_local_vllm_ids_with_logprobs()` in detail

[evaluation.py:775](../openrlhf/trainer/ppo_utils/evaluation.py#L775)

**Job:** "vLLM, continue each of these prompts (given as token IDs), and tell me how confident you were on every token you wrote."

### Inputs

| Argument | Meaning |
|---|---|
| `prompt_token_ids` | A list of prompts, each already a list of token IDs. The tree code builds these by hand (prompt + prefix up to the fork). |
| `llm` | Either a Ray actor wrapping vLLM (training) or a plain `vllm.LLM` (local runs). |
| `n` | Samples per prompt. Always 1 here; the tree code duplicates prompts instead (see `T` below). |
| `max_tokens`, `temperature`, `top_p`, `min_tokens` | Normal sampling settings. |
| `stops` | **Token IDs** that end generation (for example `<|im_end|>`). Passed as `stop_token_ids`. |
| `skip_special_tokens` | Whether decoded text hides special tokens. |

### What it does, step by step

1. **Build `SamplingParams`** ([line 787](../openrlhf/trainer/ppo_utils/evaluation.py#L787)) with `logprobs=True`. This is the important flag: without it there is nothing to rank fork points by.
2. **Call vLLM** ([lines 807-815](../openrlhf/trainer/ppo_utils/evaluation.py#L807-L815)):
   - In training, `llm` is a Ray actor, so it calls `ray.get(llm.generate.remote(...))`.
   - Otherwise it calls `llm.generate(...)` directly. Each prompt is wrapped in `TokensPrompt`, because newer vLLM removed the `prompt_token_ids=` argument.
3. **Unpack each result** ([lines 825-844](../openrlhf/trainer/ppo_utils/evaluation.py#L825-L844)). `out.logprobs` is a list with one entry per generated token. Each entry is a dict like `{token_id: Logprob(logprob=-0.31, ...)}`.
   - `next(iter(d.keys()))` is the **token ID** that was sampled.
   - `next(iter(d.values())).logprob` is **its log-probability**.
   - This relies on vLLM putting the sampled token *first* in that dict, which it does.
4. **Return five parallel lists**, one item per prompt:
   ```
   (token_ids, text, finish_reason, num_tokens, log_probs)
   ```
   `finish_reason` is `"stop"` (hit an EOS token) or `"length"` (ran out of `max_tokens`).
5. **Retry on errors.** It retries up to `RETRY_COUNT` (10) times with growing sleeps. If the sleep would pass 30 s, it calls `exit(1)`. If every retry fails, it returns `(None, None, None, None, None)`.

### Caveat

Depending on the vLLM version (and its `logprobs_mode` setting), the returned logprob may come from the **raw** model distribution, not the one after temperature and top_p. If exact values matter, check which vLLM version you run.

---

## 3. `entropy_guided_chain()` in detail

[entropy_chain_local_manager.py:154](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L154)

### The four knobs

| Knob | Meaning | Typical (`scripts/treerl-qw14b.sh`) |
|---|---|---|
| **M** | Number of trees, i.e. independent full answers to start from | 6 |
| **N** | Fork points picked **per tree** in each round | 2 |
| **L** | Number of forking rounds | 1 |
| **T** | New endings sampled at each fork point | 2 |

Each round adds `M × N × T` new branches. With (6, 2, 1, 2), that's 6 roots + 24 branches = **30 leaves** per problem.

### The data structure: `TreeNode`

[tree_node.py:29](../openrlhf/trainer/ppo_utils/tree_node.py#L29)

A `TreeNode` is **one call to vLLM**, meaning one continuous piece of generated text. It stores:

- `token_id_list`, `token_str_list`, `log_prob_list`: the text and one logprob per token.
- `parent_node`, `parent_node_split_idx`: which node it branched from, and at which token position.
- `aggregate_token_ids`: all the tokens *before* this node, from the root down (the parent's aggregate plus `parent.token_id_list[:split_idx]`).
- `mask`: one boolean per token. `True` means "never fork here". A token is masked when:
  - it is position 0 of a child (that token was *just* re-sampled at the fork);
  - it would go past `max_length`;
  - it comes after a token whose text contains `"answer"` or `"conclusion"` (forking inside the final answer is pointless).
  ([tree_node.py:92-113](../openrlhf/trainer/ppo_utils/tree_node.py#L92-L113))
- `binary_score`, `score`: filled in later by grading.

`self.tree_lists` is a list of M lists. `tree_lists[i]` holds every `TreeNode` of tree *i*, with parents always before children.

### Step by step

**Step 1: encode the prompt** ([line 178](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L178)). `encode_fn` applies the chat template and returns token IDs, `init_prompt_ids_with_template`.

**Step 2: grow M roots** ([lines 196-221](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L196-L221)).
- Send the same prompt M times to `query_local_vllm_ids_with_logprobs` (up to 4 attempts).
- Each full answer becomes the root `TreeNode` of its own tree.

**Step 3: repeat L rounds** ([line 224](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L224)).

*3a. Pick fork points* ([lines 229-292](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L229-L292)). For each tree:
1. For every node in the tree (roots *and* earlier branches) that still has unmasked tokens, call `node.get_max_entropy_tokens(top_n=N)`. This returns that node's N highest-surprisal unmasked positions.
2. Pool the candidates from all nodes of the tree into one list of `(surprisal, tree_idx, node_idx, node, token_idx)`.
3. Sort by surprisal, highest first, and keep the **top N for the whole tree**. N is per tree, not per node.

There are two alternative ways to pick:
- `use_diverse_sampling`: take `N × diverse_upsampling` candidates, then [`select_diverse_tokens`](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L468) greedily picks N that are "far apart". Note that "distance" here is `abs(token_id_a - token_id_b)`, the gap between *vocabulary IDs*, which says little about meaning.
- `random_fork` (our B1 ablation): pick N random unmasked positions instead.

*3b. Build prompts for the fork points* ([lines 299-305](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L299-L305)).
```
prompt = init_prompt_ids + node.get_prefix_ids(split_idx)
       = prompt + (all ancestor tokens) + node.token_id_list[:split_idx]
```
The token **at** `split_idx` is left out, so the model re-samples exactly the token it was unsure about. `expansion_tasks * T` repeats the list T times, so each fork gets T independent endings. `task_mapping[i]` remembers which fork each prompt belongs to.

*3c. Generate all endings in one batch* ([line 308](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L308)). This is the same vLLM helper. If it fails, the round is skipped.

*3d. Attach the new nodes* ([lines 321-342](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L321-L342)). Each ending becomes a child `TreeNode` with `parent_node_split_idx = split_idx`. It is linked with `parent.add_child(...)` and appended to its tree's list.

**Step 4: grade every node** ([`evaluate_trees`](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L123), 8 threads). For each node, [`evaluate_node`](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L82) computes:
- `binary_score`: 1 if `finish_reason == "stop"` and `check_result` says the answer matches the gold answer, otherwise 0. Truncated answers always get 0.
- `score`, depending on flags:
  - `use_pure_binary`: `score = binary_score`.
  - `use_pure_RM`: `score = sigmoid(a·(RM − b))`, using a remote reward model.
  - otherwise: `score = binary + 0.5·sigmoid(RM)`.

**Step 5: turn the tree into training data** ([lines 403-426](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L403-L426)). `build_into_tree_format` and then `gather_paths`, explained in section 4.

### A tiny example (M=1, N=1, L=1, T=2)

```
root:   [Let][x][=][2][.][So][we][get][7]        surprisal of [So] = 3.1 (highest)
                          ▲ split_idx = 5
prompt for fork = problem + [Let][x][=][2][.]
child 1:                 [Thus][we][need][3]     (mask[0]=True on [Thus])
child 2:                 [Hence][x][is][7]
```

---

## 4. From tree to per-token rewards

This is what makes TreeRL different from GRPO: **different parts of the same answer get different rewards.**

### 4a. Cut nodes into segments

[`build_into_tree_format`](../openrlhf/trainer/ppo_utils/tree_node.py#L208)

A `TreeNode` holds a full continuation, but branches split it at fork positions. The code cuts every `TreeNode` at its children's split points into `MCTSNode` **segments**:

```
root TreeNode:   [Let x = 2 .] [So we get 7]
                       │             └─ segment "tail" (terminal, R = root's score)
                       └─ children: the two fork TreeNodes, each its own terminal segment
```

A root `MCTSNode` with empty text sits on top and holds all M trees. Every `TreeNode` ends in exactly one terminal segment, so **#leaves = #TreeNodes**.

### 4b. Normalize the leaves and push values up

[`leaf_normalize`](../openrlhf/trainer/ppo_utils/tree_node.py#L377)

1. **Leave-one-out baseline:** `leaf.R = R − mean(R of all other leaves)`.
2. Walk up from every leaf. Every ancestor gets `terminal_in_subtree += 1` and `accumulated_value += leaf.R`.
3. So `V(segment) = accumulated_value / terminal_in_subtree`, the average normalized reward of all leaves below that segment.
4. Optionally, [`normalize_all_steps`](../openrlhf/trainer/ppo_utils/tree_node.py#L421) subtracts a global mean (the `overall_norm_style` flag).

### 4c. Pick leaves and weight them

- [`select_terminal`](../openrlhf/trainer/ppo_utils/tree_node.py#L483) picks `num_traces` leaves. It makes sure at least one correct leaf is in (or a balanced mix, with `balance_ratio`).
- With `use_weighted_value` and the `sqrt` style, [`compute_weighted_update`](../openrlhf/trainer/ppo_utils/tree_node.py#L553) divides each segment's value by `sqrt(#selected leaves below it)`. Otherwise a prefix shared by many leaves would be counted many times over.

### 4d. Walk each selected leaf's path

[`gather_paths`](../openrlhf/trainer/ppo_utils/parallel_mcts.py#L1622) builds one path per selected leaf, from root to leaf, using [`path_from_root_to_node`](../openrlhf/trainer/ppo_utils/parallel_mcts.py#L1606). For every segment on the path it records:

- `state_value = V(segment)`: how good this state is overall (global advantage).
- `value = V(segment) − V(parent)`: whether this segment made things better or worse (local advantage).
- `pass_ratio`: the fraction of correct leaves below.

With `use_state_value_reward` (the TreeRL script's setting), the final reward is `value + state_value`, matching the paper's *global + local advantage*.

### Worked example: why shared prefixes get 0

One tree, with root answer ✓ and one fork ✗:

```
leaves:        tail ✓ (R=1),   fork ✗ (R=0)
leave-one-out: tail = 1 − 0 = +1,   fork = 0 − 1 = −1
prefix "Let x = 2 ."  →  V = (+1 + −1)/2 = 0     ← led to one right, one wrong: neutral
"So we get 7"         →  V = +1 → reward > 0     ← pushed toward the right answer
"Thus we need 3"      →  V = −1 → reward < 0     ← pushed toward the wrong answer
```

The model is taught that the fork token was where things went right or wrong, and the shared start is neither praised nor blamed.

---

## 5. The whole pipeline

### 5a. In RL training (the original TreeRL use)

```
scripts/treerl-qw14b.sh
  └─ train_reinforce_ray.py  (--use_mcts --use_entropy_tree --m 6 --n 2 --l 1 --t 2 ...)
      └─ ReinforceActor → trainer.fit(...)                      reinforce_actor.py:357 passes m,n,l,t,...
          └─ RemoteExperienceMakerReinforce.make_experience      experience_maker_reinforce.py:140
              └─ sample_responses_bymcts                         experience_maker.py:2056
                  │  (one problem at a time)
                  └─ _generate_vllm_mcts                         experience_maker.py:3105
                      │  picks a vLLM engine (round-robin by rank), builds the args dict
                      └─ parallel_entropy_guided_tree            entropy_guided_tree_search.py:70
                          └─ EntropyGuidedChainLocalManager.process_single_item
                              └─ entropy_guided_chain        ← builds + grades the tree
                                  ├─ query_local_vllm_ids_with_logprobs   (vLLM, logprobs)
                                  ├─ evaluate_trees                        (grading)
                                  └─ build_into_tree_format + gather_paths (per-segment values)
                      ◄── paths: num_traces lists of segments {token_answer, value, pass_ratio, ...}
                      │
                      └─ back in _generate_vllm_mcts (≈ lines 3270-3346):
                           - each path → one sequence = prompt_ids + concatenated segment tokens
                           - each segment's value is copied onto every token in it → token-level rewards
                           - pad/truncate to generate_max_len, record overlong
                           - bonus: if pass@1 < correct_bonus_threshold, scale correct paths' rewards
              └─ reference model logprobs → KL penalty
                 compute_reward_naive(..., process_reward=True)   models/utils.py:40
              └─ REINFORCE-style policy update (no critic)
```

Links:
- [reinforce_actor.py:357](../openrlhf/trainer/ray/reinforce_actor.py#L357)
- [experience_maker_reinforce.py:140](../openrlhf/trainer/ppo_utils/experience_maker_reinforce.py#L140)
- [experience_maker.py:2056](../openrlhf/trainer/ppo_utils/experience_maker.py#L2056), [:3105](../openrlhf/trainer/ppo_utils/experience_maker.py#L3105), [:3210](../openrlhf/trainer/ppo_utils/experience_maker.py#L3210)
- [entropy_guided_tree_search.py:70](../openrlhf/trainer/ppo_utils/entropy_guided_tree_search.py#L70)
- [models/utils.py:40](../openrlhf/models/utils.py#L40)

**Key point:** the logprobs that vLLM returns are used **only to choose fork points**. The training loss uses logprobs that the actor and reference models recompute on the final sequences, not the vLLM ones.

### 5b. In our inference-only runs (`mid_submission/run_treerl.py`)

We skip the whole RL side and call the tree builder directly.

```
run_treerl.py main()
  ├─ VLLMGenerator(model, temperature, top_p, ...)      plain vllm.LLM (no Ray)
  ├─ treerl_args(a, eos)                                 mirrors the non-GLM args of _generate_vllm_mcts
  │     method b0 → L=0 (M independent chains, no forks)
  │     method b1 → random_fork=True (random fork points)
  │     method b2 → normal EPTree (surprisal fork points)
  └─ per problem:
       mgr = EntropyGuidedChainLocalManager(args, gen.llm, ...)
       mgr.entropy_guided_chain(question, answer, args)        ← same function as in training
       to_tree(..., mgr.tree_lists, ...)                        read TreeRL's TreeNodes into our Tree
       score(t, ...)                                            our own verifier grades each leaf
       append_jsonl(out, t.to_json())
```

Links: [run_treerl.py:36](run_treerl.py#L36) (`treerl_args`), [:54](run_treerl.py#L54) (`to_tree`), [:125-127](run_treerl.py#L125-L127) (the call).

Here `evaluator_urls` is empty and `use_pure_binary=True`, so TreeRL's own grading is only a record (`treerl_grade`). Our [`Verifier`](tree_alloc/verify.py) produces the reward we actually use. The steps in section 4 (per-segment values) still run inside `entropy_guided_chain`, but we don't use their output.

---

## 6. Things worth knowing (gotchas)

1. **"Entropy" is surprisal of the sampled token**, not distribution entropy (section 1).
2. **N is per tree, per round**, pooled across all of that tree's nodes. It is not N per node.
3. **A forked position is not masked afterwards.** When L ≥ 2, the same parent position can be picked again in a later round, which gives more siblings at the same spot. With L=1 this doesn't matter.
4. **Duplicate forks are possible.** If a node has fewer than `top_n` unmasked tokens, `get_max_entropy_tokens` fills the list by repeating indices ([tree_node.py:203-204](../openrlhf/trainer/ppo_utils/tree_node.py#L203-L204)), so one position can be expanded twice.
5. **`diverse_sampling` measures distance with token IDs**, which is a weak proxy for "different".
6. **Masking uses substrings:** any token whose text contains "answer" or "conclusion" blocks forking from there to the end of that node.
7. **Truncated answers (`finish_reason == "length"`) always score 0**, even if they contain the right number.
8. **Temperature matters twice.** It sets how random the endings are, *and* whether vLLM's reported logprobs are raw or post-temperature (depends on the vLLM version). Both change which positions look "surprising".
