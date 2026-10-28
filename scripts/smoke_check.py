"""Checks for scripts/smoke_test.sh: read one smoke run's outputs, print PASS / WARN / FAIL lines.

  python scripts/smoke_check.py train <method> <run dir> <stdout log> <exit code> <steps> <prompts> "<M N L T>" <KL>
  python scripts/smoke_check.py eval <summary csv> <per-problem jsonl> <exit code> <benchmarks> <problems each>

method: treerl | chainrl | grpo. Exit status 1 if any check failed (WARN does not fail).
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys

failed = False


def check(ok, what, warn=False):
    global failed
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    failed |= not ok and not warn
    print(f"   {tag}  {what}")
    return ok


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def train(method, run_dir, log_path, rc, steps, prompts, tree, kl):
    m, n, l, t = map(int, tree.split())
    leaves = m * (1 + n * l * t)
    log = open(log_path, errors="replace").read() if os.path.exists(log_path) else ""
    check(rc == 0, f"training exited with status {rc}")
    check(os.path.exists(os.path.join(run_dir, "model.safetensors")), "final model saved")
    for bad in ("tree_record failed", "node_summary failed", "tree log failed"):
        check(bad not in log, f'no "{bad}" in the log')
    check("use all terminals into training" in log, "trains on all leaves (--use_all_terminals)")
    if kl > 0:
        check("ReferenceModelRayActor" in log, f"KL {kl}: the reference model ran")

    steps_log = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))
    if not check(len(steps_log) == steps, f"train_log.jsonl: {len(steps_log)} of {steps} steps"):
        return
    keys = ["policy_loss", "grad_norm", "kl", "lr", "train_time", "rollout_time", "gen_tokens", "trained_tokens",
            "trained_responses", "generated_responses", "adv_mean", "adv_abs", "resumed_from", "pass_at_1"]
    missing = sorted({k for s in steps_log for k in keys if k not in s})
    check(not missing, f"every step logs {', '.join(keys)}" + (f" (missing: {missing})" if missing else ""))
    per_step = prompts * leaves
    check(all(s.get("trained_responses") == per_step == s.get("generated_responses") for s in steps_log),
          f"{per_step} responses generated and trained per step ({prompts} prompts x {leaves})")
    check(any(s.get("grad_norm", 0) > 0 for s in steps_log), "non-zero gradient")
    check(all(math.isfinite(s.get("kl", float("nan"))) and math.isfinite(s.get("policy_loss", float("nan")))
              for s in steps_log), "loss and KL finite")
    if kl > 0 and steps >= 3:  # step 1 trains at lr 0 (warmup), so step 3 is the first rollout of a changed policy
        check(abs(steps_log[-1]["kl"]) > 0, f"KL after the policy changed: {steps_log[-1]['kl']:.2e}", warn=True)
    check(all(s.get("resumed_from") == 0 for s in steps_log), "not resumed")

    trees = read_jsonl(os.path.join(run_dir, "trees.jsonl"))
    if not check(len(trees) == steps * prompts, f"trees.jsonl: {len(trees)} of {steps * prompts} prompt records"):
        return
    check(all(r["leaves"] == r["trained"] == leaves and len(r["nodes"]) == m for r in trees),
          f"every record: {leaves} leaves, all trained, {m} trees")
    forks = [sum(nd["parent"] is not None for tr in r["nodes"] for nd in tr) for r in trees]
    check(all(f == leaves - m for f in forks), f"{leaves - m} forked responses per prompt")
    check(all(nd["fork_nll"] is not None for r in trees for tr in r["nodes"] for nd in tr if nd["parent"] is not None),
          "fork surprisal recorded for every fork")
    if n:
        check(all(r["gen_tokens"] < r["trained_tokens"] for r in trees), "trees: shared prefixes generated once")
    else:
        check(all(r["gen_tokens"] == r["trained_tokens"] for r in trees), "chains: generated = trained tokens")
        check(all(len(p["segs"]) == 1 for r in trees for p in r["paths"]), "chains: one segment per response")
        means = [statistics.mean(p["segs"][0][1] for p in r["paths"]) for r in trees]
        check(all(abs(x) < 1e-3 for x in means), "advantages centred per prompt (mean 0)")
        if method == "grpo":
            check(all(len({p["segs"][0][1] for p in r["paths"]}) <= 2 for r in trees),
                  "GRPO: one advantage for correct and one for wrong responses")

    times = [(s["rollout_time"] + s["train_time"]) / 60 for s in steps_log[2:]] or \
            [(s["rollout_time"] + s["train_time"]) / 60 for s in steps_log]
    per = statistics.median(times)
    print(f"   info  {per:.2f} min per step at this batch (median of steps {3 if len(steps_log) > 2 else 1}+); "
          f"generated tokens per prompt {statistics.mean(r['gen_tokens'] for r in trees):.0f}")
    if method == "treerl" and prompts == 16 and leaves == 30:  # the timing run
        print(f"   info  TreeRL at the paper batch: {per:.2f} min/step: plan the runs with it (RERUN_PAPER.md section 6)")


def evaluate(summary, per_problem, rc, benchmarks, each):
    check(rc == 0, f"evaluation exited with status {rc}")
    rows = list(csv.DictReader(open(summary))) if os.path.exists(summary) else []
    check(len(rows) == benchmarks and all(int(r["problems"]) == each for r in rows),
          f"summary: {len(rows)} of {benchmarks} benchmarks, {each} problems each")
    pp = read_jsonl(per_problem)
    check(len(pp) == benchmarks * each and all({"correct", "n", "len", "trunc"} <= set(r) for r in pp),
          f"per-problem file: {len(pp)} of {benchmarks * each} rows with correct, n, len, trunc")
    for r in rows:
        print(f"   info  {r['data']}: accuracy {float(r['mean_acc']):.2f} on {r['problems']} problems")


if __name__ == "__main__":
    kind, a = sys.argv[1], sys.argv[2:]
    if kind == "train":
        train(a[0], a[1], a[2], int(a[3]), int(a[4]), int(a[5]), a[6], float(a[7]))
    elif kind == "eval":
        evaluate(a[0], a[1], int(a[2]), int(a[3]), int(a[4]))
    else:
        sys.exit(f"unknown check {kind}")
    sys.exit(1 if failed else 0)
