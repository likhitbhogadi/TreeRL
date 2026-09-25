"""CLI. Every command writes/reads JSONL (one tree or oracle record per line) and
resumes by skipping problem ids already present in --out.

  gen      build trees with b0 | b1 | b2 | p1, then score leaves
  p2       spend k extra rollouts on scored P1 trees (one variant per call)
  oracle   Task 5 oracle rollouts at candidate states of scored P1 trees
  summary  Phase-I style table over tree logs
  report   Phase-II / oracle table (expected oracle variance per variant, Spearman,
           value error), with paired bootstrap CIs against P2-rand

Examples (see mid_submission/tier1_implementation.md for the full sweep):
  python -m tree_alloc.run gen --method b0 --n 16 --data D --out logs/b0.jsonl
  python -m tree_alloc.run gen --method p1 --budget 12 --lam 1 --data D --out logs/p1.jsonl
  python -m tree_alloc.run p2 --from logs/p1.jsonl --variant absdelta --k 4 --out logs/p2_abs.jsonl
  python -m tree_alloc.run oracle --from logs/p1.jsonl --limit 40 --out logs/oracle.jsonl
  python -m tree_alloc.run report --p1 logs/p1.jsonl --oracle logs/oracle.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, Iterable, List

from . import methods as M
from .data import load_problems
from .gen import FakeGenerator, VLLMGenerator
from .metrics import mean, paired_bootstrap, spearman, tree_metrics
from .oracle import candidate_states, expected_score, run_oracle
from .segments import SegTree
from .tree import Tree
from .verify import Verifier

DEFAULT_VARIANTS = [("rand", 0.5, "all"), ("unif", 0.5, "all"), ("delta", 0.5, "all"),
                    ("absdelta", 0.5, "all"), ("delta", 0.5, "branch_children"),
                    ("absdelta", 0.5, "branch_children"), ("spread", 0.25, "all")]


# ---------------------------------------------------------------- io
def read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def make_backend(a):
    if a.backend == "fake":
        return FakeGenerator(seed=a.seed), Verifier("string")
    gen = VLLMGenerator(a.model, a.temperature, a.top_p, a.max_model_len, a.seed, a.gpu_mem)
    return gen, Verifier(a.verifier)


def rng_for(seed, *parts) -> random.Random:
    return random.Random("|".join(map(str, (seed,) + parts)))


# ---------------------------------------------------------------- commands
def cmd_gen(a):
    gen, ver = make_backend(a)
    problems = load_problems(a.data)
    if a.select_from:  # keep problems whose B0 accuracy is in [acc_min, acc_max]
        acc = {r["problem_id"]: mean([n["reward"] for n in r["nodes"]]) for r in read_jsonl(a.select_from)}
        problems = [p for p in problems if a.acc_min <= acc.get(p.id, -1) <= a.acc_max]
    problems = problems[: a.limit] if a.limit else problems
    done = {r["problem_id"] for r in read_jsonl(a.out)}
    todo = [p for p in problems if p.id not in done]
    cfg = {k: v for k, v in vars(a).items() if k not in ("func",)}
    print(f"[gen] {a.method}: {len(todo)} problems to run ({len(done)} already in {a.out})")
    for batch in chunks(todo, a.chunk):
        t0 = time.time()
        trees = [Tree(p.id, gen.encode_prompt(p.question), p.answer, {"method": a.method, "cfg": cfg})
                 for p in batch]
        rngs = [rng_for(a.seed, a.method, t.problem_id) for t in trees]
        kw = dict(max_response=a.max_response, max_model_len=a.max_model_len)
        if a.method == "b0":
            M.run_iid(trees, gen, a.n, **kw)
        elif a.method in ("b1", "b2"):
            M.run_eptree(trees, gen, rngs, a.M, a.N, a.L, a.T, random_fork=a.method == "b1",
                         tail_frac=a.tail_frac, **kw)
        elif a.method == "p1":
            M.run_p1(trees, gen, rngs, a.budget, score=a.score, lam=a.lam, per_round=a.per_round,
                     min_suffix=a.min_suffix, root_eligible=not a.no_root, **kw)
        M.score(trees, gen, ver)
        append_jsonl(a.out, (t.to_json() for t in trees))
        print(f"[gen] {len(batch)} trees, {sum(t.new_tokens for t in trees)} new tokens, "
              f"{time.time() - t0:.0f}s")


def cmd_p2(a):
    gen, ver = make_backend(a)
    done = {r["problem_id"] for r in read_jsonl(a.out)}
    src = [r for r in read_jsonl(getattr(a, "from")) if r["problem_id"] not in done]
    for batch in chunks(src, a.chunk):
        trees = [Tree.from_json(r) for r in batch]
        for t in trees:
            t.meta["method"] = f"p2-{a.variant}"
        rngs = [rng_for(a.seed, "p2", a.variant, a.tau, a.candidates, t.problem_id) for t in trees]
        M.run_p2(trees, gen, ver, rngs, a.k, a.variant, a.max_response, a.max_model_len,
                 tau=a.tau, candidates=a.candidates)
        append_jsonl(a.out, (t.to_json() for t in trees))
        print(f"[p2] {a.variant}: {len(trees)} trees")


def parse_variants(spec: str):
    if not spec:
        return DEFAULT_VARIANTS
    out = []
    for item in spec.split(","):  # e.g. "absdelta:0.5:all,spread:0.25:all"
        v, tau, c = item.split(":")
        out.append((v, float(tau), c))
    return out


def cmd_oracle(a):
    gen, ver = make_backend(a)
    done = {r["problem_id"] for r in read_jsonl(a.out)}
    src = read_jsonl(getattr(a, "from"))
    if a.limit:
        src = src[: a.limit]
    src = [r for r in src if r["problem_id"] not in done]
    variants = parse_variants(a.variants)
    for batch in chunks(src, a.chunk):
        trees = [Tree.from_json(r) for r in batch]
        states = [candidate_states(SegTree(t), variants, a.max_states)[0] for t in trees]
        recs = run_oracle(trees, states, gen, ver, a.R, a.K, a.Mq, a.max_response, a.max_model_len)
        append_jsonl(a.out, recs)
        print(f"[oracle] {len(trees)} trees, {len(recs)} states")


def cmd_summary(a):
    cols = ["n_leaves", "new_tokens", "acc", "pass_any", "mixed", "distinct_answers",
            "n_branch_points", "mixed_branch_frac", "delta_zero_frac", "truncated_frac"]
    print("| log | problems | " + " | ".join(cols) + " | fork_rel_pos |")
    print("|" + "---|" * (len(cols) + 3))
    for path in a.logs:
        ms = [tree_metrics(Tree.from_json(r)) for r in read_jsonl(path)]
        if not ms:
            continue
        row = [f"{mean([m.get(c) for m in ms]):.3g}" for c in cols]
        pos = [p for m in ms for p in m["fork_rel_pos"]]
        print(f"| {os.path.basename(path)} | {len(ms)} | " + " | ".join(row) + f" | {mean(pos):.2f} |")


def _find_state(seg: SegTree, key: str):
    for v in seg.nodes:
        if "%d:%d" % seg.state_key(v) == key:
            return v
    return None


def cmd_report(a):
    trees = {r["problem_id"]: Tree.from_json(r) for r in read_jsonl(a.p1)}
    recs = read_jsonl(a.oracle)
    by_problem: Dict[str, Dict[str, Dict]] = {}
    for r in recs:
        by_problem.setdefault(r["problem_id"], {})[r["state_key"]] = r
    pids = [p for p in by_problem if p in trees]
    variants = parse_variants(a.variants)

    print(f"## Expected oracle variance at allocated states ({len(pids)} problems)\n")
    print("| variant | var_total | var_between | covered mass | var_between - rand [95% CI] |")
    print("|---|---|---|---|---|")
    per_var: Dict[str, Dict[str, Dict[str, float]]] = {}
    for v, tau, c in variants:
        name = f"{v}|{tau}|{c}"
        per_var[name] = {"var_total": {}, "var_between": {}, "mass": {}}
        for pid in pids:
            seg = SegTree(trees[pid])
            dist = {"%d:%d" % seg.state_key(seg.nodes[s]): p
                    for s, p in M.state_distribution(seg, v, tau, c).items()}
            for metric in ("var_total", "var_between"):
                val, mass = expected_score(dist, by_problem[pid], metric)
                per_var[name][metric][pid] = val
            per_var[name]["mass"][pid] = mass
    base = per_var.get("rand|0.5|all")
    for name, d in per_var.items():
        ci = ""
        if base is not None and name != "rand|0.5|all":
            b = paired_bootstrap(d["var_between"], base["var_between"])
            ci = f"{b['diff']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}]"
        print(f"| {name} | {mean(d['var_total'].values()):.4f} | {mean(d['var_between'].values()):.4f} "
              f"| {mean(d['mass'].values()):.2f} | {ci} |")

    # How well do small-tree signals predict the oracle? (Spearman over states)
    sig = {"max|delta| of children": [], "child-value spread": [], "var_between": [], "var_total": []}
    err: Dict[str, List[float]] = {"1": [], "2-3": [], "4+": []}
    for pid in pids:
        seg = SegTree(trees[pid])
        for key, r in by_problem[pid].items():
            u = _find_state(seg, key)
            if u is None:
                continue
            n = len(u.leaves)
            err["1" if n == 1 else "2-3" if n <= 3 else "4+"].append(abs(seg.value(u) - r["p_hat"]))
            if r["var_between"] is None or not u.children:
                continue
            vals = [seg.value(c) for c in u.children]
            sig["max|delta| of children"].append(max(abs(x - seg.value(u)) for x in vals))
            sig["child-value spread"].append(max(vals) - min(vals))
            sig["var_between"].append(r["var_between"])
            sig["var_total"].append(r["var_total"])
    print(f"\n## Spearman: tree signal vs oracle ({len(sig['var_between'])} states)\n")
    print("| signal | vs var_between | vs var_total |\n|---|---|---|")
    for s in ("max|delta| of children", "child-value spread"):
        print(f"| {s} | {spearman(sig[s], sig['var_between']):.3f} | {spearman(sig[s], sig['var_total']):.3f} |")
    print("\n## |V_tree - p_hat| by #leaves under the state (Task 6)\n")
    print("| leaves | n | mean abs error |\n|---|---|---|")
    for k, xs in err.items():
        print(f"| {k} | {len(xs)} | {mean(xs):.3f} |")

    for path in a.trees or []:  # Task 6 across methods: value error of final trees at oracle states
        errs = []
        for r in read_jsonl(path):
            if r["problem_id"] not in by_problem:
                continue
            seg = SegTree(Tree.from_json(r))
            for key, o in by_problem[r["problem_id"]].items():
                u = _find_state(seg, key)
                if u is not None:
                    errs.append(abs(seg.value(u) - o["p_hat"]))
        print(f"\n{os.path.basename(path)}: mean |V - p_hat| = {mean(errs):.3f} over {len(errs)} states")


# ---------------------------------------------------------------- argparse
def main(argv=None):
    ap = argparse.ArgumentParser(prog="tree_alloc")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def backend_args(p):
        p.add_argument("--backend", choices=["vllm", "fake"], default="vllm")
        p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
        p.add_argument("--verifier", choices=["auto", "math_verify", "string"], default="auto")
        # 1.0, not TreeRL's 1.2: at 1.2 Qwen2.5-Math-1.5B collapses into token soup in ~13%
        # of samples (smoke test), which also dominates surprisal-based forking.
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--top_p", type=float, default=0.95)
        p.add_argument("--max_model_len", type=int, default=4096)
        p.add_argument("--max_response", type=int, default=3072)
        p.add_argument("--gpu_mem", type=float, default=0.9)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--chunk", type=int, default=256, help="problems per lockstep batch")
        p.add_argument("--out", required=True)

    g = sub.add_parser("gen")
    backend_args(g)
    g.add_argument("--method", choices=["b0", "b1", "b2", "p1"], required=True)
    g.add_argument("--data", required=True)
    g.add_argument("--limit", type=int)
    g.add_argument("--select_from", help="B0 log used to keep problems by accuracy")
    g.add_argument("--acc_min", type=float, default=0.1)
    g.add_argument("--acc_max", type=float, default=0.9)
    g.add_argument("--n", type=int, default=16, help="b0: chains per problem")
    g.add_argument("--M", type=int, default=6)
    g.add_argument("--N", type=int, default=2)
    g.add_argument("--L", type=int, default=1)
    g.add_argument("--T", type=int, default=2)
    g.add_argument("--tail_frac", type=float, default=0.1)
    g.add_argument("--budget", type=int, default=12, help="p1: n-k trajectories")
    g.add_argument("--score", choices=list(M.P1_SCORES), default="mu_sigma")
    g.add_argument("--lam", type=float, default=1.0)
    g.add_argument("--per_round", type=int, default=1)
    g.add_argument("--min_suffix", type=int, default=32)
    g.add_argument("--no_root", action="store_true", help="p1: never branch from the prompt")
    g.set_defaults(func=cmd_gen)

    p = sub.add_parser("p2")
    backend_args(p)
    p.add_argument("--from", required=True, help="scored P1 log")
    p.add_argument("--variant", choices=list(M.P2_VARIANTS), required=True)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--candidates", choices=["all", "branch_children"], default="all")
    p.set_defaults(func=cmd_p2)

    o = sub.add_parser("oracle")
    backend_args(o)
    o.add_argument("--from", required=True, help="scored P1 log")
    o.add_argument("--limit", type=int, default=40)
    o.add_argument("--max_states", type=int, default=30)
    o.add_argument("--R", type=int, default=32, help="full rollouts per state")
    o.add_argument("--K", type=int, default=8, help="next-step samples per state (0 = skip)")
    o.add_argument("--Mq", type=int, default=4, help="rollouts per next step")
    o.add_argument("--variants", default="", help="v:tau:cands,... (default: all)")
    o.set_defaults(func=cmd_oracle)

    s = sub.add_parser("summary")
    s.add_argument("logs", nargs="+")
    s.set_defaults(func=cmd_summary)

    r = sub.add_parser("report")
    r.add_argument("--p1", required=True)
    r.add_argument("--oracle", required=True)
    r.add_argument("--trees", nargs="*", help="final tree logs to score against the oracle (Task 6)")
    r.add_argument("--variants", default="")
    r.set_defaults(func=cmd_report)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
