"""RL results -> CSV tables (from results/rl/, as copied from the server's TreeRL_rl/).

  python -m tree_alloc.rl_report results/rl

Reads eval.csv (rows appended by solve_rate.py --summary), train_logs/<run>.jsonl and gpu_mem/<run>.log. Writes:
  eval_table.csv    one row per (run, step, decoding): accuracy on MATH500 / AMC / Omni-MATH, pass@8 when sampled
  train_curves.csv  one row per (run, step); for a resumed run the last attempt's line for a step wins
  runs.csv          per run: steps logged, peak GPU memory of our processes (MiB)
  paired.csv        if per_problem/*.jsonl exist (solve_rate.py --out): each model's accuracy minus the base model's
                    on the same problems, with a 95% paired-bootstrap CI over problems (as Task 2's Table 2);
                    data=all pools the three benchmarks (1,082 problems)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

from .metrics import paired_bootstrap

DATA = {"MATH500.jsonl": "math500", "aimo-validation-amc.jsonl": "amc", "omni_math_500_seed0.jsonl": "omni"}
TRAIN_COLS = ["pass_at_1", "pass_rate", "reward", "policy_loss", "grad_norm", "response_length", "response_entropy",
              "response_overlong_ratio", "generate_time", "rollout_time"]


def run_step(model):
    if not model.startswith("/"):
        return "base", 0
    m = re.search(r"ckpt/([^/]+)/_actor_global_step(\d+)", model)
    return m.group(1), int(m.group(2))


def main(d):
    table = {}
    for r in csv.DictReader(open(os.path.join(d, "eval.csv"))):
        run, step = run_step(r["model"])
        key = (run, step, "greedy" if r["n"] == "1" else f"mean@{r['n']} T={r['temperature']}")
        row = table.setdefault(key, {"run": key[0], "step": key[1], "decoding": key[2]})
        row[DATA[r["data"]]] = round(100 * float(r["mean_acc"]), 2)
        if r["n"] != "1":
            row[f"{DATA[r['data']]}_pass@{r['n']}"] = round(100 * float(r["pass_at_n"]), 2)
    cols = ["run", "step", "decoding", "math500", "amc", "omni", "math500_pass@8", "amc_pass@8", "omni_pass@8"]
    with open(os.path.join(d, "eval_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(sorted(table.values(), key=lambda r: (r["decoding"], r["run"] != "base", r["run"], r["step"])))

    runs = []
    with open(os.path.join(d, "train_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["run", "step"] + TRAIN_COLS)
        w.writeheader()
        for path in sorted(glob.glob(os.path.join(d, "train_logs", "*.jsonl"))):
            run = os.path.basename(path)[:-len(".jsonl")]
            by_step = {}
            lines = [json.loads(l) for l in open(path) if l.strip()]
            for x in lines:
                by_step[x["global_step"]] = x
            for s in sorted(by_step):
                w.writerow({"run": run, "step": s, **{c: by_step[s].get(c) for c in TRAIN_COLS}})
            mem = os.path.join(d, "gpu_mem", run + ".log")
            peak = max((int(l.split()[1]) for l in open(mem) if len(l.split()) == 2), default=None) \
                if os.path.exists(mem) else None
            runs.append({"run": run, "steps_logged": len(by_step), "max_step": max(by_step, default=0),
                         "log_lines": len(lines), "peak_gpu_mem_mib": peak})
    with open(os.path.join(d, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    pp = {}  # name -> {benchmark: {problem id: accuracy over its n samples}}
    for path in glob.glob(os.path.join(d, "per_problem", "*.jsonl")):
        name = os.path.basename(path)[:-len(".jsonl")].removesuffix("_pp")
        for line in open(path):
            r = json.loads(line)
            bench = pp.setdefault(name, {}).setdefault(DATA[r["data"]], {})
            # key by position + id: problems are written in file order, and AMC's ids repeat (two contests)
            bench[f"{len(bench)}:{r['id']}"] = r["correct"] / r["n"]
    if "base_n8" in pp:
        with open(os.path.join(d, "paired.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "data", "problems", "acc", "base_acc", "diff", "ci_lo", "ci_hi"])
            for name in sorted(pp):
                for bench in ("math500", "amc", "omni", "all"):  # all: pooled over the three benchmarks' problems
                    pool = lambda m: {f"{k}/{i}": v for k, d in pp[m].items() for i, v in d.items()}  # noqa: E731
                    a, b = (pool(name), pool("base_n8")) if bench == "all" else (pp[name].get(bench), pp["base_n8"].get(bench))
                    if name == "base_n8" or not a or not b:
                        continue
                    bs = paired_bootstrap(a, b)
                    w.writerow([name, bench, bs["n"], round(100 * sum(a.values()) / len(a), 2),
                                round(100 * sum(b.values()) / len(b), 2),
                                *(round(100 * bs[k], 2) for k in ("diff", "lo", "hi"))])
        print(f"wrote paired.csv ({len(pp) - 1} models vs base)")
    print(f"wrote eval_table.csv ({len(table)} rows), train_curves.csv, runs.csv ({len(runs)} runs) in {d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/rl")
