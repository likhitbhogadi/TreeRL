"""Sample n answers per problem with vLLM and grade them. Two uses:

  # RL training data: keep the train_30k problems the base model solves sometimes but not always
  # (a tree whose leaves are all right or all wrong gets zero advantage, i.e. no gradient, in TreeRL)
  python solve_rate.py --data ../datasets/train/train_30k.jsonl --sample 3000 --n 8 \
      --keep_mixed data/train_30k_mixed.jsonl

  # evaluate a model or an RL checkpoint: greedy accuracy (or e.g. --n 4 --temperature 1.0: mean over 4 samples)
  python solve_rate.py --model ../ckpt/<tag>/_actor_global_step40 --temperature 0 --summary results/rl_eval.csv \
      --data ../datasets/eval/MATH500.jsonl ../datasets/eval/aimo-validation-amc.jsonl data/omni_math_500_seed0.jsonl

Prompts are the RL prompts (Qwen2.5-Math chat template and its "reason step by step ... \\boxed{}" system prompt);
problems with prompts over 1024 tokens are skipped, as RL's --prompt_max_len would truncate them. A truncated
response counts as wrong. Per-problem correct counts go to --out (JSONL).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tree_alloc.data import load_problems  # noqa: E402
from tree_alloc.verify import Verifier  # noqa: E402

MAX_PROMPT = 1024  # RL's --prompt_max_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--sample", type=int, help="random subset of this many problems from each file")
    ap.add_argument("--n", type=int, default=8, help="samples per problem (1 when --temperature 0)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--gpu_mem", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="per-problem results (JSONL)")
    ap.add_argument("--keep_mixed", help="write the rows solved 0 < c < n times here, with a solve_rate field")
    ap.add_argument("--summary", help="append one CSV row per data file")
    a = ap.parse_args()

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from tree_alloc.gen import VLLMGenerator

    gen = VLLMGenerator(a.model, max_model_len=MAX_PROMPT + a.max_tokens, seed=a.seed, gpu_memory_utilization=a.gpu_mem)
    ver = Verifier()
    n = 1 if a.temperature == 0 else a.n
    sp = SamplingParams(n=n, temperature=a.temperature, top_p=a.top_p if a.temperature else 1.0,
                        max_tokens=a.max_tokens, seed=a.seed)
    mixed = open(a.keep_mixed, "w") if a.keep_mixed else None

    for path in a.data:
        probs = load_problems(path)
        if a.sample:
            random.Random(a.seed).shuffle(probs)
            probs = probs[:a.sample]
        prompts = [gen.encode_prompt(p.question) for p in probs]
        keep = [i for i, ids in enumerate(prompts) if len(ids) <= MAX_PROMPT]
        t0 = time.time()
        outs = gen.llm.generate([TokensPrompt(prompt_token_ids=prompts[i]) for i in keep], sp)
        rows = []
        for i, o in zip(keep, outs):
            p = probs[i]
            c = sum(s.finish_reason == "stop" and ver(s.text, p.answer) > 0 for s in o.outputs)
            rows.append({"data": os.path.basename(path), "id": p.id, "n": n, "correct": c})
            if mixed and 0 < c < n:
                mixed.write(json.dumps({**p.raw, "solve_rate": c / n}) + "\n")

        acc = sum(r["correct"] for r in rows) / (n * len(rows))
        pass_n = sum(r["correct"] > 0 for r in rows) / len(rows)
        n_mixed = sum(0 < r["correct"] < n for r in rows)
        print(f"{os.path.basename(path)}: {len(rows)} problems ({len(probs) - len(rows)} skipped, prompt > "
              f"{MAX_PROMPT} tokens) | mean acc {acc:.3f} | pass@{n} {pass_n:.3f} | mixed {n_mixed} | "
              f"{time.time() - t0:.0f}s", flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
        if a.summary:
            new = not os.path.exists(a.summary)
            with open(a.summary, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["model", "data", "problems", "n", "temperature", "mean_acc", "pass_at_n"])
                w.writerow([a.model, os.path.basename(path), len(rows), n, a.temperature, f"{acc:.4f}", f"{pass_n:.4f}"])
    if mixed:
        mixed.close()


if __name__ == "__main__":
    main()
