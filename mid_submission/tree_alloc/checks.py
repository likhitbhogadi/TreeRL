"""Task 1 smoke-test checks on a real-model log (run on the GPU box after `gen`):

  python -m tree_alloc.checks --log logs/smoke_b0.jsonl

1. logprobs are raw pi_theta: re-score logged sequences with a transformers forward pass and
   compare against log_softmax(logits) and the *processed* distribution vLLM samples from
   (logits / T, then top-p renormalized). At T = 1.0 only the top-p part differs, so it is included.
2. segmentation: segments per response, and a response printed with "‖" at each boundary.
3. verifier: extracted answer vs gold for a sample, for eyeballing.
"""
from __future__ import annotations

import argparse
import random

from .run import read_jsonl
from .tree import Tree


def processed_logprobs(logits, temperature, top_p):
    """log p of each token under softmax(logits / T) restricted to the top-p nucleus and renormalized."""
    import torch

    probs = torch.softmax(logits / temperature, -1)
    sp, idx = probs.sort(-1, descending=True)
    keep = sp.cumsum(-1) - sp < top_p  # smallest prefix whose mass reaches top_p
    mask = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, idx, keep)
    p = torch.where(mask, probs, torch.zeros_like(probs))
    return torch.log(p / p.sum(-1, keepdim=True))


def check_logprobs(trees, model_name, temperature, top_p, n_seqs, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device).eval()
    rows = []
    for t in trees[:n_seqs]:
        ids, lps = t.response(0)
        x = torch.tensor([t.prompt_ids + ids], device=device)
        with torch.no_grad():
            logits = model(x).logits[0, len(t.prompt_ids) - 1:-1].float()
        tgt = torch.tensor(ids, device=device)
        raw = torch.log_softmax(logits, -1).gather(1, tgt[:, None])[:, 0]
        proc = processed_logprobs(logits, temperature, top_p).gather(1, tgt[:, None])[:, 0]
        v = torch.tensor(lps, device=device)
        rows.append(((v - raw).abs().mean().item(), (v - proc).abs().mean().item(), len(ids)))
    print("\n## 1. Are logged logprobs raw pi_theta?")
    print(f"| seq | tokens | mean |vllm - raw| | mean |vllm - processed (T={temperature}, top-p={top_p})| |"
          "\n|---|---|---|---|")
    for i, (a, b, n) in enumerate(rows):
        print(f"| {i} | {n} | {a:.4f} | {b:.4f} |")
    raw_wins = sum(a < b for a, b, _ in rows)
    print(f"\n=> closer to RAW in {raw_wins}/{len(rows)} sequences "
          "(bf16 noise of ~0.01-0.05 is expected; the other column should be clearly larger)")


def check_segments(trees, tok):
    import statistics

    n_segs, lens = [], []
    for t in trees:
        for n in t.nodes:
            b = t.boundaries(n.id)
            n_segs.append(len(b))
            prev = 0
            for e in b:
                lens.append(e - prev)
                prev = e
    print("\n## 2. Segmentation")
    print(f"segments/response: mean {statistics.mean(n_segs):.1f}, median {statistics.median(n_segs)}, "
          f"max {max(n_segs)}; tokens/segment: median {statistics.median(lens)}, max {max(lens)}")
    t = trees[0]
    ids, _ = t.response(0)
    cuts = set(t.boundaries(0))
    text = "".join(tok.decode([i]) + ("‖" if k + 1 in cuts else "") for k, i in enumerate(ids))
    print("\nexample (‖ = segment boundary):\n" + text[:3000])


def check_verifier(trees, tok, n):
    print("\n## 3. Verifier sample (check by hand)")
    print("| problem | gold | extracted | reward | finish |\n|---|---|---|---|---|")
    rng = random.Random(0)
    leaves = [(t, nd) for t in trees for nd in t.nodes]
    for t, nd in rng.sample(leaves, min(n, len(leaves))):
        g = str(t.gold).replace("|", "\\|")
        a = str(nd.answer).replace("|", "\\|")
        print(f"| {t.problem_id[:30]} | {g[:40]} | {a[:40]} | {nd.reward} | {nd.finish_reason} |")
    acc = sum(nd.reward for _, nd in leaves) / len(leaves)
    trunc = sum(nd.finish_reason == "length" for _, nd in leaves) / len(leaves)
    none = sum(nd.answer is None for _, nd in leaves) / len(leaves)
    print(f"\naccuracy {acc:.3f}, truncated {trunc:.3f}, no \\boxed answer {none:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--n_logprob_seqs", type=int, default=4)
    ap.add_argument("--n_verify", type=int, default=30)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer

    trees = [Tree.from_json(r) for r in read_jsonl(a.log)]
    tok = AutoTokenizer.from_pretrained(a.model)
    check_segments(trees, tok)
    check_verifier(trees, tok, a.n_verify)
    check_logprobs(trees, a.model, a.temperature, a.top_p, a.n_logprob_seqs)


if __name__ == "__main__":
    main()
