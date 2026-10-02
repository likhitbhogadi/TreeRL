# Code walkthrough: Tasks 1 and 2, step by step

This explains how the Task 1 and Task 2 code works, in the order it runs. Every file and function name is a link: click it to open the code.

- **Task 1 (the harness):** build trees of model responses to math problems, and save every detail to disk.
- **Task 2 (replicate EPTree):** build trees with different sampling methods, then compare them the way the TreeRL paper does.

**Who does what.** TreeRL's own code builds the trees. Our code (`tree_alloc/` and two scripts) handles everything around that: loading problems and the model, grading answers, saving trees, and all the analysis.

---

## 0. Five ideas you need first

**1. A response is a list of tokens, and each token has a probability.** When the model writes "Let x = 2", it produces tokens one at a time. For each token it also reports how likely it thought that token was. We store the **log-probability** (a negative number: close to 0 means "very sure", very negative means "surprised").

**2. Surprisal = how surprised the model was by its own token** = −log-probability.
- A token with probability 0.9 has surprisal 0.1.
- A token with probability 0.02 has surprisal 3.9.

EPTree forks (branches) where surprisal is high: places where the model was unsure what to write.

**3. A tree of responses.** Instead of writing 16 separate answers from scratch, you can:
1. write a few full answers;
2. go back to a point in one of them;
3. write a new ending from there.

The new answer **reuses the start** of the old one, which saves tokens:
```
chain:   Let x = 2 . So we get ── 7   ✓     (original answer)
                        └─ fork ── we need ── 3   ✗   (new branch: same start, new ending)
```
Every finished answer (every path to an end) is a **leaf**.

**4. (M, N, L, T) = the four EPTree settings.**

| Letter | Meaning | Example (6,2,1,2) |
|---|---|---|
| **M** | how many full answers to write first | 6 chains |
| **N** | how many fork points to pick in each tree, per round | 2 per chain |
| **L** | how many rounds of forking | 1 round |
| **T** | how many new branches from each fork point | 2 each |

Leaves = M + L·M·N·T. For (6,2,1,2) that's 6 + 1·6·2·2 = **30 answers**.

**5. PassRate = % of problems where at least one answer is correct.** This is the paper's main metric. More, and more diverse, answers means more chances to get one right.

**The three methods compared in Task 2:**

| Code name | Method | How it picks fork points |
|---|---|---|
| **b0** | i.i.d. chains | No forking: just M independent answers (TreeRL writes this as (M,0,0,0)) |
| **b1** | random forks | N random token positions |
| **b2** | EPTree | The N tokens with the highest surprisal |

---

## 1. The big picture

```
                         ┌──────────────── run_treerl.py (one process per shard) ───────────────┐
 problems (JSONL) ──► load_problems ──► for each problem:                                       │
                         │   1. TreeRL builds the tree  (EntropyGuidedChainLocalManager)        │
                         │   2. convert to our Tree     (to_tree)                               │
                         │   3. grade every leaf        (score + Verifier)                      │
                         │   4. append one JSON line    (append_jsonl)                          │
                         └──────────────────────────────────────────────────────────────────────┘
                                         │  logs/task2/<config>.jsonl  (one tree per line)
                                         ▼
                  task2_report.py ──► Figs 4/5/7/8 + Table 2
```

`task2.sh` runs this for all 11 configs, in parallel across GPU processes, and then runs the report.

---

## 2. Task 1: building and saving one tree, step by step

### Step 1: Load the problems — [tree_alloc/data.py](tree_alloc/data.py)

[`load_problems`](tree_alloc/data.py) reads a `.jsonl` file (one problem per line) and returns `Problem(id, question, answer)` objects.
- Datasets name their fields differently: `problem` / `question` / `Question`, and `answer` / `final_answer` / `Answer`. The loader tries each name.
- OlympiadBench stores answers like `"['2']"`, so `_answer` unwraps that to `2`.

### Step 2: Load the model — [tree_alloc/gen.py](tree_alloc/gen.py)

[`VLLMGenerator`](tree_alloc/gen.py) loads Qwen2.5-Math-1.5B-Instruct into **vLLM**, a fast inference engine. It sets up three things:
- **Raw logprobs** (`logprobs_mode="raw_logprobs"`): the probabilities come from the model itself, *not* after temperature or top-p reshaping. Surprisal has to describe the model, not the sampler.
- **Prompts as token IDs** ([`encode_prompt`](tree_alloc/gen.py)): the question is wrapped in the chat template ("system: reason step by step… user: <question> assistant:") and turned into token IDs. (Newer `transformers` versions return a dict here, which the function handles.)
- **FlashInfer sampler switched off:** on pkgpu2 it tries to compile code with a mismatched CUDA toolkit and crashes.

### Step 3: TreeRL builds the tree — [run_treerl.py](run_treerl.py) → TreeRL's code

[`main()`](run_treerl.py) in `run_treerl.py` creates TreeRL's manager for each problem and calls `entropy_guided_chain`. Settings come from [`treerl_args`](run_treerl.py), which mirrors TreeRL's own launch settings.

Inside TreeRL's [`entropy_guided_chain`](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L154):

**3a. Write M full answers** ([line 190](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L190)).
- One vLLM call generates M answers to the same prompt, using [`query_local_vllm_ids_with_logprobs`](../openrlhf/trainer/ppo_utils/evaluation.py#L775).
- Each answer becomes the root `TreeNode` of its own small tree.
- [`TreeNode`](../openrlhf/trainer/ppo_utils/tree_node.py#L28) stores the tokens, their logprobs, and where it branched from.

**3b. Decide which tokens are allowed as fork points** (the "mask", in `TreeNode.__init__`). A token can't be a fork point if:
- it's the first token of a branch ([line 96](../openrlhf/trainer/ppo_utils/tree_node.py#L96)): that would just repeat the fork;
- it's past the length limit ([line 100](../openrlhf/trainer/ppo_utils/tree_node.py#L100));
- it comes after any token containing "answer" or "conclusion" ([line 110](../openrlhf/trainer/ppo_utils/tree_node.py#L110)): don't branch inside the final answer.

**3c. Pick N fork points per tree** ([line 224](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L224) onward, once per round, L rounds):
- **b2 (EPTree):** [`get_max_entropy_tokens`](../openrlhf/trainer/ppo_utils/tree_node.py#L182) scores every allowed token by surprisal, then the N highest in the whole tree are picked ([line 260](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L260)).
- **b1 (random):** N allowed positions are picked at random ([line 230](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L230)). We added this, because the paper uses it but the released code doesn't have it.
- **b0 (i.i.d.):** L = 0, so this step never runs.

**3d. Write T new endings from each fork point** ([lines 301–339](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L301)).
- For a fork at position *t*, the prompt is `question + the answer's first t tokens`, as **token IDs** ([`get_prefix_ids`](../openrlhf/trainer/ppo_utils/tree_node.py#L160)).
- The model continues from there, and the result becomes a child `TreeNode`.
- *Why token IDs and not text?* Turning text back into tokens can split words differently at the cut point, so the branch wouldn't truly continue the same prefix.

**3e. TreeRL grades the leaves** ([line 388](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py#L388)). This is for its RL rewards, which we don't use. We made its grader work without its lab's judge servers ([`_local_check`](../openrlhf/trainer/ppo_utils/evaluation.py#L344)). It's a simple string comparison, because the symbolic checker can freeze when run from a thread.

At the end, `mgr.tree_lists` holds M lists of `TreeNode`s: one tree per initial chain.

### Step 4: Convert to our tree format — [`to_tree`](run_treerl.py) and [tree_alloc/tree.py](tree_alloc/tree.py)

TreeRL's `TreeNode`s live in memory and point at each other. For saving and analysis, we convert them into our [`Tree`](tree_alloc/tree.py), which has the same shape but plain numbers instead of pointers.

Each [`Node`](tree_alloc/tree.py) is **one generation call**:

| Field | Meaning |
|---|---|
| `parent` | which node it branched from (`None` = written from the question) |
| `fork_idx` | where in the parent it branched (a token index) |
| `offset` | where its first token sits in the full answer |
| `token_ids`, `logprobs` | what it wrote, and how sure the model was about each token |
| `finish_reason` | `stop` (finished normally) or `length` (cut off) |
| `seg_ends`, `boxed_at` | "\n\n" step boundaries, and where `\boxed{` starts (from [`annotate`](tree_alloc/tree.py)) |
| `reward`, `answer` | correct (1) or not (0), and the extracted answer |

Useful `Tree` methods:
- [`response(id)`](tree_alloc/tree.py) glues a node's full answer back together: its ancestors' prefixes plus its own tokens.
- [`prefix_ids`](tree_alloc/tree.py) gives the prefix a branch continued from.
- [`new_tokens`](tree_alloc/tree.py) counts only tokens actually generated. Shared prefixes are counted once, which is exactly the token saving trees give you.

**Worked example.** One chain of 13 tokens, with a fork at token 4:
```
node 0 (root):   [Let  x  .  \n\n | so  we  .  \n\n  \  boxed  {  7  }]      offset 0, 13 tokens
node 1 (fork):   parent 0, fork_idx 4  → [get  .  \n\n  \  boxed  {  3  }]      offset 4, 8 tokens
full answer of node 1 = node 0's first 4 tokens + node 1's 8 tokens = 12 tokens
new_tokens = 13 + 8 = 21   (the 4 shared tokens are counted once)
```

### Step 5: Grade every answer ourselves — [`score`](run_treerl.py) and [tree_alloc/verify.py](tree_alloc/verify.py)

For each leaf:
1. rebuild its full text;
2. [`extract_boxed`](tree_alloc/verify.py) pulls out what's inside the last `\boxed{...}`, matching braces;
3. [`Verifier`](tree_alloc/verify.py) compares it with the gold answer using `math_verify`, which knows that `\frac{1}{2}` = `0.5`, with a normalized string match as backup.

A cut-off answer (`finish_reason == "length"`) scores 0. This runs in the main thread, where `math_verify`'s time limit works, so it can't freeze.

### Step 6: Save one line per tree — [tree_alloc/run.py](tree_alloc/run.py)

- [`append_jsonl`](tree_alloc/run.py) writes the tree as one JSON line to `logs/task2/<config>.jsonl`.
- **Resuming:** on start, `run_treerl.py` reads that file and skips problems already done. A stopped run continues where it left off.
- **Parallel shards:** with `--shard i --num_shards K`, a process only takes every K-th problem and writes to `<config>.jsonl.shard<i>`. That lets several copies of TreeRL's one-problem-at-a-time code run side by side.

---

## 3. Task 2: the experiment and its analysis, step by step

### Step 1: Run every config — [task2.sh](task2.sh)

`task2.sh` lists 11 configs:
- i.i.d. with 64 chains;
- EPTree and random at (8,4,2,2), (6,2,1,2), (4,3,1,1) and (4,1,1,1);
- EPTree only at (2,3,1,1) and (2,1,1,1).

It then does four things:
1. **Makes a to-do list** (`fill_queue`): every (config, shard) pair that isn't finished yet. Each config is split into K = 8 shards of about 63 problems.
2. **Starts workers** (`worker`): SLOTS processes per GPU, each taking the next item from the list. Before starting, a worker checks the GPU has enough **free memory** (`fits`), because the machine is shared with other users. If a job fails, it goes back on the list once, and the worker pauses first.
3. **Runs a second pass** for anything that failed or was cut short.
4. **Merges** the 8 shard files into `<config>.jsonl`, but only if all 500 problems are there. Then it runs the report.

### Step 2: The main report — [tree_alloc/task2_report.py](tree_alloc/task2_report.py)

[`main()`](tree_alloc/task2_report.py) loads every config's trees and keeps only problems present in all of them, so the comparisons are fair. Then:

**a. The i.i.d. baseline curve** ([`iid_curve`](tree_alloc/task2_report.py)). From the 64-chain run, it computes "if I had drawn only k answers, what's the chance at least one is right?" for k = 1…64.
- It doesn't just use the first k answers, which would be noisy. It uses the exact **unbiased pass@k formula** ([`pass_at_k`](tree_alloc/task2_report.py)):
  pass@k = 1 − C(n−c, k) / C(n, k), with n answers of which c are correct
  = 1 − the chance that k random picks are all wrong.
- The token cost of k chains is the total length of that problem's first k answers.

**b. Per-config numbers** ([`tree_stats`](tree_alloc/task2_report.py)): PassRate, average accuracy, tokens used, number of leaves, number of **distinct final answers** (a diversity measure), and whether each tree had both right and wrong answers.

**c. Table 2: entropy vs. random** at the same (M,N,L,T), with a **paired bootstrap** 95% confidence interval ([`paired_bootstrap`](tree_alloc/metrics.py)):
- Resample the 500 problems with replacement 2,000 times, recomputing the difference each time.
- The middle 95% of those differences is the interval.
- If it includes 0, the difference could be luck.

**d. The figures** ([`plot_all`](tree_alloc/task2_report.py)):

| Figure | Shows | Question it answers |
|---|---|---|
| Fig. 5 | PassRate vs. tokens spent; the i.i.d. curve plus a dot per tree config | Do trees get more right answers for the same cost? |
| Fig. 4 | Answers and distinct answers vs. tokens | Do trees give more (and more varied) answers per token? |
| Fig. 7 | Top-10 tokens where EPTree forks ([`fork_info`](tree_alloc/task2_report.py)) | What does "uncertain" look like? |
| Fig. 8 | Where forks happen (fork position ÷ answer length) | Early, late, or everywhere? |

([`place_labels`](tree_alloc/task2_report.py) just keeps figure labels from overlapping.)

### Step 3: A quick table for any log — [tree_alloc/run.py](tree_alloc/run.py) and [tree_alloc/metrics.py](tree_alloc/metrics.py)

`python -m tree_alloc.run logs/task2/*.jsonl` prints one row per config, using [`tree_metrics`](tree_alloc/metrics.py): leaves, tokens, accuracy, PassRate, distinct answers, truncated answers, and average fork position.

---

## 4. File map

| File | Task | One-line purpose |
|---|---|---|
| [run_treerl.py](run_treerl.py) | 1–2 | Runs TreeRL's tree builder per problem, converts, grades, saves |
| [task2.sh](task2.sh) | 2 | Runs all 11 configs in parallel shards, merges, reports |
| [tree_alloc/data.py](tree_alloc/data.py) | 1 | Load problems from any of the datasets |
| [tree_alloc/gen.py](tree_alloc/gen.py) | 1 | Load the model in vLLM (raw logprobs, chat template) |
| [tree_alloc/tree.py](tree_alloc/tree.py) | 1 | Our tree format: nodes, prefixes, boundaries, JSON |
| [tree_alloc/verify.py](tree_alloc/verify.py) | 1 | Extract `\boxed{}` answers and grade them |
| [tree_alloc/run.py](tree_alloc/run.py) | 1–2 | Read/write JSONL, summary table |
| [tree_alloc/metrics.py](tree_alloc/metrics.py) | 2 | Per-tree metrics, paired bootstrap |
| [tree_alloc/task2_report.py](tree_alloc/task2_report.py) | 2 | pass@k, Table 2, Figs 4/5/7/8 |
| TreeRL: [entropy_chain_local_manager.py](../openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py) | 1–2 | Builds the tree (steps 3a–3e) |
| TreeRL: [tree_node.py](../openrlhf/trainer/ppo_utils/tree_node.py) | 1–2 | `TreeNode`, fork masking, top-surprisal picking |
| TreeRL: [evaluation.py](../openrlhf/trainer/ppo_utils/evaluation.py) | 1–2 | vLLM calls and grading (our edits make them run without Ray or judge servers) |

---

## 5. Common questions

**Why temperature 1.0 and not the paper's 1.2?** At 1.2 this small model writes random gibberish in 13% of answers. Gibberish tokens also have the highest surprisal, so EPTree would fork right inside it.

**Why do we re-grade answers when TreeRL already grades them?** TreeRL's grader normally calls LLM servers in the authors' lab. Our local stand-in is a simple string check, so it can't freeze inside TreeRL's threads. The real grading, which all results use, is our `Verifier`.

**Why shards?** TreeRL's code handles one problem at a time, which leaves the GPU mostly idle. Running 3–4 copies per GPU on different problems keeps it busy.

**Why save everything to JSONL instead of just the final numbers?** Every analysis (figures, tables) is computed afterwards from the saved trees. A new question never needs a new GPU run.

**What should I run to see it working?**
```bash
python run_treerl.py --method b2 --M 2 --N 1 --L 1 --T 1 \
    --data ../datasets/eval/MATH500.jsonl --limit 5 --out logs/try.jsonl   # 5 small trees, on pkgpu2
python -m tree_alloc.run logs/try.jsonl                                  # summary table
python -m tree_alloc.run logs/task2/*.jsonl                              # summary of the Task 2 runs
```
