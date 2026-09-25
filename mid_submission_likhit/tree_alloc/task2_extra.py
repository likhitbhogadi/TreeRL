"""Task 2 add-on analyses (offline, from the same logs as task2_report):

  python -m tree_alloc.task2_extra --logs logs/task2 --out results/task2

A. Tree vs. i.i.d. with paired bootstrap CIs, matched per problem at the same #leaves and at the
   same generated tokens (i.i.d. cost for problem p at k chains = k * mean chain length of p;
   unbiased pass@k from the 64-chain B0 run, linearly interpolated in k).
B. Sibling disagreement: for each fork, does the new branch's outcome differ from the original
   continuation it forked from (the parent's own leaf)? Reference: two i.i.d. chains disagree
   with probability 2c(n-c)/(n(n-1)).
C. Continuation length of each fork vs. the remainder of the original it replaced.
D. Estimators of V(root) vs. the policy's true pass@1 (from B0): leaf-mean (the proposal's
   V(v), eq. 2), child-mean (each child subtree weighted equally, recursively over the segment
   tree), and roots-only (mean of the M initial chains -- unbiased by construction).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics as S
from typing import Dict, List

from .metrics import paired_bootstrap
from .run import read_jsonl
from .segments import SegTree
from .task2_report import pass_at_k
from .tree import Tree


def load(path: str) -> Dict[str, Tree]:
    return {r["problem_id"]: Tree.from_json(r) for r in read_jsonl(path)}


def iid_pass_interp(n: int, c: int, k: float) -> float:
    k = max(1.0, min(k, float(n)))
    lo, hi = int(k), min(int(k) + 1, n)
    f = k - lo
    return (1 - f) * pass_at_k(n, c, lo) + f * pass_at_k(n, c, hi)


def child_mean(seg: SegTree) -> float:
    """Bottom-up without recursion (some degenerate responses have >1000 segments); children are
    always created after their parent, so reverse creation order is a valid post-order."""
    val: Dict[int, float] = {}
    for u in reversed(seg.nodes):
        vals = [val[c.idx] for c in u.children] + [seg.reward(l) for l in u.ends]
        val[u.idx] = sum(vals) / len(vals)
    return val[seg.root.idx]


def ci(b) -> str:
    return f"{b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]"


def fmt_pct_ci(b) -> str:
    return f"{100 * b['diff']:+.1f} [{100 * b['lo']:+.1f}, {100 * b['hi']:+.1f}]"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    b0 = load(os.path.join(a.logs, "b0_64.jsonl"))
    runs = {}
    for path in sorted(glob.glob(os.path.join(a.logs, "b[12]_*.jsonl"))):
        m = re.match(r"(b[12])_(\d+-\d+-\d+-\d+)\.jsonl", os.path.basename(path))
        if m:
            runs[(m.group(1), m.group(2))] = load(path)
    pids = sorted(set(b0).intersection(*[set(t) for t in runs.values()]))
    n0 = len(b0[pids[0]].nodes)
    c = {p: sum(nd.reward > 0 for nd in b0[p].nodes) for p in pids}
    chain_len = {p: S.mean(len(nd.token_ids) for nd in b0[p].nodes) for p in pids}
    pass1 = {p: c[p] / n0 for p in pids}
    name = lambda k, cfg: f"{'EPTree' if k == 'b2' else 'Random'} ({cfg.replace('-', ',')})"  # noqa: E731
    order = sorted(runs, key=lambda kc: (S.mean(t.new_tokens for t in runs[kc].values()), kc[0]))

    L = [f"# Task 2 add-on analyses\n", f"{len(pids)} problems; i.i.d. reference = {n0} chains/problem.\n"]

    # ---- A
    L.append("## A. Tree vs. i.i.d., paired per problem (95% bootstrap CI)\n")
    L.append("PassRate difference in points. *Same leaves*: i.i.d. pass@k with k = #leaves (only if ≤ "
             f"{n0}). *Same tokens*: i.i.d. pass@k at the k whose cost equals the tree's tokens on that problem.\n")
    L.append("| method | leaves | tree PassRate | Δ vs i.i.d. same leaves | Δ vs i.i.d. same tokens | "
             "matched k (mean) |\n|---|---|---|---|---|---|")
    for key in order:
        trees = runs[key]
        tree_pass = {p: float(max(nd.reward for nd in trees[p].nodes) > 0) for p in pids}
        leaves = S.mean(len(trees[p].nodes) for p in pids)
        same_leaves = "n/a (> %d)" % n0
        if leaves <= n0:
            iid = {p: pass_at_k(n0, c[p], len(trees[p].nodes)) for p in pids}
            same_leaves = fmt_pct_ci(paired_bootstrap(tree_pass, iid))
        ks = {p: trees[p].new_tokens / chain_len[p] for p in pids}
        iid_t = {p: iid_pass_interp(n0, c[p], ks[p]) for p in pids}
        L.append(f"| {name(*key)} | {leaves:.0f} | {100 * S.mean(tree_pass.values()):.1f}% | {same_leaves} | "
                 f"{fmt_pct_ci(paired_bootstrap(tree_pass, iid_t))} | {S.mean(ks.values()):.1f} |")

    # ---- B + C
    ref = S.mean(2 * c[p] * (n0 - c[p]) / (n0 * (n0 - 1)) for p in pids)
    L.append("\n## B. Sibling disagreement and C. continuation length, per fork\n")
    L.append(f"Reference: two independent chains from the prompt disagree in outcome **{100 * ref:.1f}%** of the time.\n")
    L.append("| method | forks | outcome differs from original | wrong→right | right→wrong | "
             "new branch tokens | original remainder | median new/remainder | new branch truncated |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    dis_by_method = {}
    for key in order:
        trees = runs[key]
        dis, w2r, r2w, new_len, rem, ratio, trunc = [], [], [], [], [], [], []
        per_problem = {}
        for p in pids:
            t = trees[p]
            d_p = []
            for nd in t.nodes:
                if nd.parent is None:
                    continue
                par = t.nodes[nd.parent]
                d = float(nd.reward != par.reward)
                dis.append(d)
                d_p.append(d)
                (w2r if par.reward == 0 else r2w).append(float(nd.reward != par.reward))
                r = len(par.token_ids) - nd.fork_idx
                new_len.append(len(nd.token_ids))
                rem.append(r)
                ratio.append(len(nd.token_ids) / max(r, 1))
                trunc.append(float(nd.finish_reason == "length"))
            if d_p:
                per_problem[p] = S.mean(d_p)
        dis_by_method[key] = per_problem
        L.append(f"| {name(*key)} | {len(dis)} | {100 * S.mean(dis):.1f}% | {100 * S.mean(w2r):.1f}% | "
                 f"{100 * S.mean(r2w):.1f}% | {S.mean(new_len):.0f} | {S.mean(rem):.0f} | "
                 f"{S.median(ratio):.2f} | {100 * S.mean(trunc):.1f}% |")
    L.append("\n*wrong→right* = share of forks off a wrong original that end correct; *right→wrong* = the reverse.\n")

    # position-stratified: an earlier fork shares less prefix, so it disagrees more regardless of
    # how the point was chosen. Compare entropy vs random forks within position bins.
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    strat = {"b2": [[] for _ in bins[:-1]], "b1": [[] for _ in bins[:-1]]}
    twins = [cfg for (k, cfg) in runs if k == "b2" and ("b1", cfg) in runs]
    for kind in ("b2", "b1"):
        for cfg in twins:
            trees = runs[(kind, cfg)]
            for p in pids:
                t = trees[p]
                for nd in t.nodes:
                    if nd.parent is None:
                        continue
                    par = t.nodes[nd.parent]
                    rel = nd.offset / par.end
                    b = next(i for i in range(len(bins) - 1) if rel < bins[i + 1])
                    strat[kind][b].append(float(nd.reward != par.reward))
    if twins:
        L.append(f"**Disagreement by fork position** (pooled over the configs run with both methods: "
                 f"{', '.join('(' + c.replace('-', ',') + ')' for c in twins)}):\n")
        L.append("| relative fork position | EPTree | Random | EPTree share of its forks | Random share of its forks |")
        L.append("|---|---|---|---|---|")
        n2, n1 = sum(map(len, strat["b2"])), sum(map(len, strat["b1"]))
        for i in range(len(bins) - 1):
            e, r = strat["b2"][i], strat["b1"][i]
            L.append(f"| {bins[i]:.1f}–{min(bins[i + 1], 1.0):.1f} | "
                     f"{100 * S.mean(e):.1f}% (n={len(e)}) | {100 * S.mean(r):.1f}% (n={len(r)}) | "
                     f"{100 * len(e) / n2:.0f}% | {100 * len(r) / n1:.0f}% |" if e and r else
                     f"| {bins[i]:.1f}–{min(bins[i + 1], 1.0):.1f} | – | – | – | – |")
        # random reweighted to EPTree's position mix = what random would score with EPTree's positions
        w = [len(e) / n2 for e in strat["b2"]]
        rew = sum(wi * S.mean(r) for wi, r in zip(w, strat["b1"]) if r)
        L.append(f"\nRandom forks reweighted to EPTree's position mix: **{100 * rew:.1f}%** vs. EPTree "
                 f"**{100 * S.mean([x for e in strat['b2'] for x in e]):.1f}%** "
                 f"(raw random {100 * S.mean([x for r in strat['b1'] for x in r]):.1f}%).\n")
    L.append("**Entropy vs. random forks, paired per problem (disagreement rate, points):**\n")
    L.append("| config | EPTree − Random [95% CI] |\n|---|---|")
    for (k, cfg) in sorted(runs):
        if k == "b2" and ("b1", cfg) in runs:
            b = paired_bootstrap(dis_by_method[("b2", cfg)], dis_by_method[("b1", cfg)])
            L.append(f"| ({cfg.replace('-', ',')}) | {fmt_pct_ci(b)} |")

    # ---- D
    L.append("\n## D. Estimating V(root) vs. the policy's pass@1\n")
    L.append(f"True pass@1 from {n0} i.i.d. chains: **{S.mean(pass1.values()):.3f}**. Bias = mean(estimate − pass@1) "
             "over problems, paired bootstrap CI; MAE = mean |estimate − pass@1|.\n")
    L.append("| method | leaf-mean bias | child-mean bias | roots-only bias | leaf-mean MAE | child-mean MAE | "
             "roots-only MAE |\n|---|---|---|---|---|---|---|")
    for key in order:
        trees = runs[key]
        est = {"leaf": {}, "child": {}, "roots": {}}
        for p in pids:
            t = trees[p]
            est["leaf"][p] = S.mean(nd.reward for nd in t.nodes)
            est["roots"][p] = S.mean(nd.reward for nd in t.nodes if nd.parent is None)
            est["child"][p] = child_mean(SegTree(t))
        bias = {e: paired_bootstrap(est[e], pass1) for e in est}
        mae = {e: S.mean(abs(est[e][p] - pass1[p]) for p in pids) for e in est}
        L.append(f"| {name(*key)} | {ci(bias['leaf'])} | {ci(bias['child'])} | {ci(bias['roots'])} | "
                 f"{mae['leaf']:.3f} | {mae['child']:.3f} | {mae['roots']:.3f} |")

    with open(os.path.join(a.out, "extra.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
