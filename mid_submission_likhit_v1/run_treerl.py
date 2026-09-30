"""Tasks 1-2 tree generation with TreeRL's own code
(openrlhf/trainer/ppo_utils/entropy_chain_local_manager.py), writing our JSONL tree logs.

  b0 = TreeRL's (M,0,0,0) i.i.d. chains
  b1 = random-forking ablation (`random_fork`, added to their manager: not in the released code)
  b2 = EPTree

  python run_treerl.py --method b2 --M 6 --N 2 --L 1 --T 2 --data data/omni_math_500_seed0.jsonl \
      --out logs/task2/b2_6-2-1-2.jsonl
  # their code handles one problem at a time, so run several shards in parallel and concatenate:
  python run_treerl.py ... --shard 0 --num_shards 4 --gpu_mem 0.2     # writes <out>.shard0

Their modules are imported through their own `except: from tree_node import ...` fallback, so the
openrlhf package (DeepSpeed, the training stack) is never imported. Every leaf is re-graded with
our verifier; TreeRL's own local grade is kept in meta["treerl_grade"].
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "openrlhf", "trainer", "ppo_utils"))
sys.path.insert(0, HERE)

from tree_alloc.data import load_problems  # noqa: E402
from tree_alloc.run import append_jsonl, read_jsonl  # noqa: E402
from tree_alloc.tree import Generation, Tree  # noqa: E402
from tree_alloc.verify import Verifier, extract_boxed  # noqa: E402


def treerl_args(a, eos):
    # mirrors the non-GLM branch of _generate_vllm_mcts (experience_maker.py:3170) with the
    # flags of scripts/treerl-qw14b.sh; post-processing flags only affect RL rewards, not the tree
    L = 0 if a.method == "b0" else a.L  # (M,0,0,0) = i.i.d. chains, as in the paper
    return {
        "random_fork": a.method == "b1",
        "temperature": a.temperature, "top_p": a.top_p, "m": a.M, "n": a.N, "l": L, "t": a.T,
        "generate_max_len": a.max_response, "evaluator_urls": [], "extractor_urls": [],
        "entropy_rm_urls": [], "eos_tokens": eos, "num_traces": a.M + L * a.M * a.N * a.T,
        "use_pure_binary": True, "use_pure_RM": False, "use_orm_reward": False, "use_chain_reward": False,
        "step_level_norm": False, "use_state_value_reward": True, "use_value_only": False,
        "balance_ratio": 0, "average_one_generation": False, "advantage_mix_allancestor": False,
        "use_weighted_value": True, "use_all_terminals": False, "a": 0.5, "b": -2.898,
        "weighted_value_style": "sqrt", "overall_norm_style": "none", "inner_repetition_penalty": False,
        "use_diverse_sampling": False, "diverse_upsampling": 1, "training_type": "math",
    }


def to_tree(problem, prompt_ids, tree_lists, token_text, meta):
    """TreeRL's per-chain TreeNode lists -> (our Tree, TreeRL's own grade per node).
    Parents always precede children in each list."""
    t = Tree(problem.id, prompt_ids, problem.answer, meta)
    ids, grades = {}, []
    for tree_list in tree_lists:
        for nd in tree_list:
            parent = ids[id(nd.parent_node)] if nd.parent_node is not None else None
            fork = nd.parent_node_split_idx if parent is not None else 0
            fin = "length" if nd.finish_reason == "length" else "stop"
            new = t.add(parent, fork, Generation(list(nd.token_id_list), list(nd.log_prob_list), fin),
                        token_text, phase="treerl")
            grades.append(int(nd.binary_score or 0))
            ids[id(nd)] = new.id
    return t, grades


def score(t: Tree, decode, verifier) -> None:
    """Binary reward per leaf; truncated responses get 0 (as in TreeRL)."""
    for n in t.nodes:
        ids, _ = t.response(n.id)
        text = decode(ids)
        n.answer = extract_boxed(text)
        n.reward = verifier(text, t.gold) if n.finish_reason == "stop" else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["b0", "b1", "b2"], required=True)
    for k, v in (("M", 6), ("N", 2), ("L", 1), ("T", 2)):
        ap.add_argument(f"--{k}", type=int, default=v)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    # 1.0, not TreeRL's 1.2: at 1.2 Qwen2.5-Math-1.5B collapses into token soup in ~13% of samples
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_response", type=int, default=3072)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = a.out if a.num_shards == 1 else f"{a.out}.shard{a.shard}"
    done = {r["problem_id"] for r in read_jsonl(out)}
    problems = [p for p in load_problems(a.data, a.limit)[a.shard::a.num_shards] if p.id not in done]
    print(f"[treerl] {a.method} shard {a.shard}/{a.num_shards}: {len(problems)} to run ({len(done)} done)", flush=True)
    if not problems:
        return

    import torch
    from entropy_chain_local_manager import EntropyGuidedChainLocalManager
    from tree_alloc.gen import VLLMGenerator

    random.seed(a.seed * 1000 + a.shard)  # b1's fork positions
    gen = VLLMGenerator(a.model, a.temperature, a.top_p, a.max_model_len, a.seed * 1000 + a.shard, a.gpu_mem)
    tok = gen.tok
    eos = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})  # ids, not strings
    encode_fn = lambda texts, max_len, device="cpu", system_prompt=None: {  # noqa: E731
        "input_ids": torch.tensor([gen.encode_prompt(texts[0][0])])}
    decode_fn = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731
    args = treerl_args(a, eos)
    ver = Verifier()
    meta = {"method": a.method, "code": "treerl", "cfg": {k: v for k, v in vars(a).items()}}

    for p in problems:
        t0 = time.time()
        try:
            mgr = EntropyGuidedChainLocalManager(args, gen.llm, encode_fn, decode_fn, [], [], eos)
            mgr.entropy_guided_chain(p.question, p.answer, args=args)
            t, grades = to_tree(p, gen.encode_prompt(p.question), mgr.tree_lists, gen.token_text, meta)
        except Exception:  # one bad problem must not kill the shard; a rerun retries it
            traceback.print_exc()
            print(f"[treerl] {p.id}: FAILED, skipped (rerun to retry)", flush=True)
            continue
        score(t, gen.decode, ver)
        t.meta = dict(meta, treerl_grade=grades)
        append_jsonl(out, [t.to_json()])
        print(f"[treerl] {p.id}: {len(t.nodes)} leaves, {t.new_tokens} tokens, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
