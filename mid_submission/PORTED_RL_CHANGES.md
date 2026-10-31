# RL changes ported onto the `adithya` branch

Branch `adithya-rl` is `origin/adithya` plus four commits. They carry over the changes from Sanyam's paper-setting reruns (branch `sanyam`) that your method needs to run the same way as TreeRL, ChainRL and GRPO. Nothing else was merged.

Your files are unchanged: `credit_aware_tree.py`, the Phase II flags and their checks, the KL change, and `logprobs_mode="raw_logprobs"`. That last setting is vLLM 0.30's default anyway, so TreeRL behaves the same on both branches.

## The four commits

| Commit | Needed? | Changes results? | What it does |
|---|---|---|---|
| `998f2b8` RL fixes | **Required** | Warmup: yes | **Warmup:** the 5% learning-rate warmup counted the wrong unit and was 0 steps at batch 480; it is now 10 steps for a 200-step run, as in the baselines. **KL reference model:** after a resume it reloaded the latest checkpoint; it now always loads the starting model. It also releases its cached GPU memory after each pass; without that, KL = 1e-4 runs out of memory on the RTX PRO 6000 (we hit this). |
| `66d6972` Launcher | **Required** to run like the baselines | No | `ADAM_OFFLOAD=` (Adam on the GPU, 1.4× faster per step), `VENV`, `SEED`, `--use_all_terminals` when all leaves are trained on, `RAY_ADDRESS=local`. It also refuses settings that otherwise go silently wrong: more leaves than the tree has, a log-prob batch that doesn't divide the leaf count, a data file shorter than the run. Your Phase II settings and leaf count (M + L·M·N·T + k) are kept. |
| `969dfac` Logging | Recommended | No | `trees.jsonl` (every tree: its nodes, with Phase II's rollouts marked `"phase": "II"`, Phase II's `phase2_log`, and each trained response's per-segment advantages). More per-step fields in `train_log.jsonl`, including `gen_tokens`, the inference cost: the same-budget comparison with TreeRL needs it. A logging error can't stop a run. |
| `e257d7f` Tooling | Recommended | No | `run_paper.sh` with a new method `ours`; cloud setup; smoke test with automatic checks (including `ours`); the auto-pause watcher; evaluation and report scripts (OlympiadBench, test problems excluded from training data). |

**Not ported:** the write-ups on `sanyam` (`RERUN_PAPER.md`, `RERUN_CODE_CHANGES.md`, `RUN_LOG.md`), and `run_baselines.sh` (the old pipeline).

## Checks done before handing over

- **Launch settings:** TreeRL, ChainRL, GRPO and `ours` each launch through the merged script, and their arguments parse and pass this branch's own validation. TreeRL, ChainRL and GRPO get exactly the settings of the runs on the instance.
- **End-to-end run on a fake LLM and grader:** `CreditAwareChainLocalManager` ran through TreeRL's advantage code and the new logging.
  - (6,2,1,2) + k = 8 gives 38 leaves, all trained; the 8 Phase II rollouts are marked, and the Phase II log has 8 entries.
  - (5,2,1,2) + k = 5 gives 30.
- **Pipeline and smoke checker:** the pipeline rehearsed with `ours` (both settings), and the smoke checker passes correct Phase II output and fails broken output.

## Two decisions before the run

**1. Rollout budget.** The default `TREE="6 2 1 2" PHASE2_K=8` generates 38 responses per prompt, while TreeRL generates 30. The proposal's claim is "without increasing the total rollout budget" (n = (n − k) + k), so this run would not be a same-budget comparison.
- `OURS_TREE="5 2 1 2" OURS_K=5` gives exactly 30 (25 from Phase I, 5 from Phase II).
- Whichever you choose, compare `gen_tokens` per step with TreeRL's.

**2. Speed and cost.** Phase II adds its k rollouts one at a time: after each one the tree is rescored before choosing the next. Each step therefore has k extra generation rounds after EPTree's 2. A round lasts as long as its longest response (~10–30 s), even though it holds only 16 sequences (one per prompt).
- Estimate: TreeRL takes 1.65 min per step on the RTX PRO 6000; `ours` probably takes ~2.5–3.5 min with k = 8, and less with k = 5.
- The timing run (`BENCH=ours`, below) measures it before committing ~10+ hours.
- If it's too slow, choosing 2–4 nodes per round would cut the rounds, but that changes the algorithm, so it's your call.

## Running it on the JarvisLabs instance

The instance already has the environment, model, training data and base-model evaluations (in `/home`). This branch goes in its own folder next to the baselines' code:

```bash
# laptop, from this worktree (TreeRL-adithya-rl); same rsync as for the baselines, into a separate folder
rsync -avz -e "ssh -p <port>" --exclude .git --exclude '*.pdf' --exclude __pycache__ \
    --exclude 'mid_submission/results' --exclude 'mid_submission_*' ./ root@<host>:/home/TreeRL-adithya/
# instance: the training data built for the baselines (same file, so the same problems)
cp /home/TreeRL/mid_submission/data/train_30k_mixed_12k.jsonl /home/TreeRL-adithya/mid_submission/data/
cd /home/TreeRL-adithya && source /home/treerl_env.sh
ONLY=ours BENCH=ours OUT=/home/smoke_ours scripts/smoke_test.sh     # ~25 min: checks, then minutes per step
ONLY=ours OURS_TREE="5 2 1 2" OURS_K=5 SAVE_STEPS=50 STEPS=200 nohup scripts/run_paper.sh > /home/paper_ours.log 2>&1 &
```

- **Shared checkpoint folder:** with `CKPT=/home/ckpt` (set by `treerl_env.sh`), the base-model evaluations already done are reused and not repeated.
- **Results land next to the baselines':** `results/rl_paper_eval.csv` and the per-problem files under `/home/TreeRL-adithya/mid_submission/`.
