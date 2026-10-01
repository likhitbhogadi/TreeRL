"""One-page status of the RL experiments. Run on pkgpu2 (no GPU needed):

  python3 ~/likhit/tree_based_rl/TreeRL_rl/scripts/status.py
  watch -n 60 python3 ~/likhit/tree_based_rl/TreeRL_rl/scripts/status.py      # refresh every minute

Reads what scripts/run_baselines.sh leaves behind: ckpt/<run>/ (checkpoints, train_log.jsonl), ckpt/.done/ (finished
evaluations), mid_submission/results/rl_eval.csv (scores), mid_submission/results/rl_per_problem/ and the pipeline log.
"""
import csv
import glob
import json
import os
import random
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT, DONE = os.path.join(ROOT, "ckpt"), os.path.join(ROOT, "ckpt", ".done")
PP = os.path.join(ROOT, "mid_submission", "results", "rl_per_problem")
LOG = "/tmp/likhit_baselines.log"
BASE = "Qwen/Qwen2.5-Math-1.5B-Instruct"
SHORT_NAME = {"MATH500.jsonl": "MATH500", "aimo-validation-amc.jsonl": "AMC", "omni_math_500_seed0.jsonl": "Omni"}
# (run tag, method, steps planned, checkpoint every, learning rate) in the order the pipeline runs them
SHORT = [("qwen1.5b-treerl-6-2-1-2", "TreeRL"), ("qwen1.5b-chainrl-8", "ChainRL"), ("qwen1.5b-grpo-8", "GRPO")]
LONG = [(t + "-lr5e-6-150", m) for t, m in SHORT]


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def steps_done(tag):
    path = os.path.join(CKPT, tag, "train_log.jsonl")
    lines = open(path).read().split("\n") if os.path.exists(path) else []
    lines = [l for l in lines if l.strip()]
    return json.loads(lines[-1])["global_step"] if lines else 0  # last line = current progress (resumes repeat steps)


def ckpts(tag):
    return sorted(int(p.rsplit("step", 1)[1]) for p in glob.glob(os.path.join(CKPT, tag, "_actor_global_step*")))


def done(name):
    return os.path.exists(os.path.join(DONE, "eval_" + name))


def per_problem(name):
    return any(os.path.exists(os.path.join(PP, n + ".jsonl")) for n in (name, name + "_pp"))


def scores():  # (model path or id, n) -> {benchmark: accuracy %}
    out, path = {}, os.path.join(ROOT, "mid_submission", "results", "rl_eval.csv")
    for r in csv.DictReader(open(path)) if os.path.exists(path) else []:
        out.setdefault((r["model"], r["n"]), {})[SHORT_NAME[r["data"]]] = 100 * float(r["mean_acc"])
    return out


def paired_diff(name, bench=None):  # bench None: all three benchmarks pooled
    """mean accuracy difference vs the base model on the same problems, with a 95% bootstrap CI (points)"""
    def load(n):
        for f in (n, n + "_pp"):
            p = os.path.join(PP, f + ".jsonl")
            if os.path.exists(p):
                rows = [r for r in (json.loads(l) for l in open(p) if l.strip()) if (bench is None or r["data"].startswith(bench))]
                return {f"{r['data']}:{i}": r["correct"] / r["n"] for i, r in enumerate(rows)}  # AMC's ids repeat
    a, b = load(name), load("base_n8")
    if not a or not b:
        return None
    d = [a[k] - b[k] for k in a if k in b]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(1000))
    return 100 * sum(d) / len(d), 100 * boots[25], 100 * boots[974]


def running():
    ps = sh("ps -u $USER -o args=")
    train = [l for l in ps.splitlines() if l.startswith("python train_reinforce_ray.py")]
    ev = [l for l in ps.splitlines() if l.startswith("python solve_rate.py")]
    tag = train[0].split("--save_path ")[1].split()[0].rsplit("/", 1)[1] if train else None
    model = ev[0].split("--model ")[1].split()[0] if ev else None
    return tag, model


def fmt(sc, keys=("MATH500", "AMC", "Omni")):
    return "  ".join(f"{k} {sc[k]:.1f}" for k in keys if k in sc) or "-"


def main():
    sc = scores()
    tag_now, eval_now = running()
    last = (open(LOG).read().strip().splitlines() or [""])[-1] if os.path.exists(LOG) else ""
    pipeline = "running" if "run_baselines.sh" in sh("ps -u $USER -o args=") else "NOT running"
    now = sh("date '+%F %T'").strip()
    print(f"=== RL experiments, Qwen2.5-Math-1.5B-Instruct === {now} (server time)")
    print(f"Pipeline (scripts/run_baselines.sh): {pipeline}. Last event: {last}")
    if tag_now:
        print(f"NOW TRAINING:   {tag_now}, step {steps_done(tag_now)}")
    if eval_now:
        print(f"NOW SCORING:    {os.path.relpath(eval_now, CKPT) if eval_now.startswith('/') else eval_now}")
    if not tag_now and not eval_now:
        print("NOW:            nothing on the GPU (waiting for free GPU memory, or finished)")

    b1, b8 = sc.get((BASE, "1"), {}), sc.get((BASE, "8"), {})
    print(f"\nBase model (before RL):  greedy {fmt(b1)}  |  8-sample mean {fmt(b8)}"
          f"  |  per-problem file for paired tests: {'yes' if per_problem('base_n8') else 'not yet'}")

    print("\nSHORT RUNS: 40 steps, lr 1.5e-6, checkpoint every 10 steps (all done in the first round)")
    for tag, method in SHORT:
        n, cs = steps_done(tag), ckpts(tag)
        last_c = os.path.join(CKPT, tag, f"_actor_global_step{cs[-1]}") if cs else None
        g = sc.get((last_c, "1"), {})
        print(f"  {method:8s} {n:3d}/40 steps   checkpoints {cs}   step {cs[-1] if cs else '-'} greedy: {fmt(g)}")

    print("\nLONG RUNS: 150 steps, lr 5e-6, checkpoint every 50 steps (scored with 8 samples per problem)")
    for tag, method in LONG:
        n, cs = steps_done(tag), ckpts(tag)
        final = os.path.exists(os.path.join(CKPT, tag, "model.safetensors"))
        state = ("done" if final else "TRAINING NOW" if tag == tag_now else
                 f"crashed at step {n}; queued to resume from its step-{cs[-1]} checkpoint" if cs else
                 "not started yet" if not n else "crashed before any checkpoint; queued to restart from 0")
        print(f"  {method:8s} {n:3d}/150 steps  [{state}]")
        for c in cs:
            name = f"{tag}__actor_global_step{c}_n8"
            s8 = sc.get((os.path.join(CKPT, tag, f"_actor_global_step{c}"), "8"), {})
            pd = paired_diff(name)
            paired = (f"vs base, same 1,082 problems: {pd[0]:+.2f} [95% CI {pd[1]:+.2f}, {pd[2]:+.2f}]" if pd else
                      "paired: not yet")
            print(f"      step {c:3d}: 8-sample {fmt(s8) if s8 else 'not scored yet'}   |   {paired}")

    print("\nPIPELINE ORDER (x = done, > = now, . = waiting)")
    stages = [
        ("Base model scored (greedy, 8-sample, per-problem)", done("base") and done("base_n8") and per_problem("base_n8")),
        ("Short runs: TreeRL, ChainRL, GRPO trained + scored", all(steps_done(t) >= 40 for t, _ in SHORT)),
        ("Long TreeRL trained (150 steps) + scored", os.path.exists(os.path.join(CKPT, LONG[0][0], "model.safetensors"))),
        ("Paired scoring (per-problem) of every long checkpoint so far",
         all(per_problem(f"{t}__actor_global_step{c}_n8") for t, _ in LONG for c in ckpts(t))),
        ("Long ChainRL trained (150 steps) + scored", os.path.exists(os.path.join(CKPT, LONG[1][0], "model.safetensors"))),
        ("Long GRPO trained (150 steps) + scored", os.path.exists(os.path.join(CKPT, LONG[2][0], "model.safetensors"))),
    ]
    first_open = next((i for i, (_, ok) in enumerate(stages) if not ok), None)
    for i, (label, ok) in enumerate(stages):
        print(f"  [{'x' if ok else '>' if i == first_open else '.'}] {label}")

    print("\nGPUs (free memory):", sh("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader").replace("\n", "  ").strip())
    print("Disk /home:", sh("df -h ~ | tail -1 | awk '{print $4\" free (\"$5\" used)\"}'").strip())


if __name__ == "__main__":
    main()
