"""Tree-construction methods. Every method advances all problems' trees in lockstep:
each round issues one batched generate() call covering every tree, so sequential
methods (P1 branches one node per round, as in Algorithm 1) stay efficient.

  B0  i.i.d. chains
  B1  random token forks, same (M, N, L, T) as EPTree
  B2  EPTree: fork at the top-N highest-surprisal tokens of each initial chain's tree
      (M + L*M*N*T leaves)
  P1  ours, Phase I: branch at the segment state with the highest S = mu + lam*sigma
  P2  ours, Phase II: k extra rollouts from parent(v), v ~ allocation distribution
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .gen import Generator
from .segments import SegNode, SegTree
from .tree import Node, Tree
from .verify import Verifier, extract_boxed


@dataclass
class Request:
    tree: int
    parent: Optional[int]
    fork_idx: int


def expand(trees: List[Tree], reqs: List[Request], gen: Generator, max_response: int,
           max_model_len: int, phase: str, round_idx: int) -> List[Node]:
    """Generate one continuation per request (single batched call) and attach it."""
    prompts, max_tok, keep = [], [], []
    for r in reqs:
        t = trees[r.tree]
        prefix = t.prefix_ids(r.parent, r.fork_idx)
        room = min(max_response - len(prefix), max_model_len - len(t.prompt_ids) - len(prefix))
        if room <= 0:
            continue
        prompts.append(t.prompt_ids + prefix)
        max_tok.append(room)
        keep.append(r)
    gens = gen.generate(prompts, max_tok) if prompts else []
    return [trees[r.tree].add(r.parent, r.fork_idx, g, gen.token_text, phase, round_idx)
            for r, g in zip(keep, gens)]


def score(trees: List[Tree], gen: Generator, verifier: Verifier) -> None:
    """Binary reward per leaf; truncated responses get 0 (as in TreeRL)."""
    for t in trees:
        for n in t.nodes:
            if n.reward is not None:
                continue
            ids, _ = t.response(n.id)
            text = gen.decode(ids)
            n.answer = extract_boxed(text)
            n.reward = verifier(text, t.gold) if n.finish_reason == "stop" else 0.0


# ---------------------------------------------------------------- B0
def run_iid(trees, gen, n, max_response, max_model_len):
    reqs = [Request(i, None, 0) for i in range(len(trees)) for _ in range(n)]
    expand(trees, reqs, gen, max_response, max_model_len, "b0", 0)


# ---------------------------------------------------------------- B1 / B2
def root_of(t: Tree, node_id: int) -> int:
    n = t.nodes[node_id]
    while n.parent is not None:
        n = t.nodes[n.parent]
    return n.id


def eptree_candidates(t: Tree, forked: set, tail_frac: float) -> Dict[int, List[Tuple[float, int, int]]]:
    """(surprisal, node id, local idx) for every token position a fork may replace,
    grouped by initial chain: TreeRL keeps one tree per chain and takes top-N in each,
    so an iteration adds M*N*T leaves (entropy_chain_local_manager.py:228-285).

    Excluded: idx 0 (would repeat the node's own fork point / a fresh chain), the
    last `tail_frac` of the trajectory, anything from "\\boxed" on, and positions
    already forked (TreeRL can re-pick them across iterations; we mask instead).
    """
    out: Dict[int, List[Tuple[float, int, int]]] = {}
    for n in t.nodes:
        group = out.setdefault(root_of(t, n.id), [])
        stop_at = (1.0 - tail_frac) * n.end
        bp = t.boxed_pos(n.id)
        for i in range(1, len(n.token_ids)):
            p = n.offset + i
            if p >= stop_at or (bp is not None and p > bp):
                break
            if (n.id, i) not in forked:
                group.append((-n.logprobs[i], n.id, i))
    return out


def run_eptree(trees, gen, rngs, M, N, L, T, max_response, max_model_len,
               random_fork=False, tail_frac=0.1):
    tag = "b1" if random_fork else "b2"
    reqs = [Request(i, None, 0) for i in range(len(trees)) for _ in range(M)]
    expand(trees, reqs, gen, max_response, max_model_len, tag, 0)
    forked = [set() for _ in trees]
    for it in range(1, L + 1):
        reqs = []
        for ti, t in enumerate(trees):
            for _, cands in sorted(eptree_candidates(t, forked[ti], tail_frac).items()):
                if random_fork:
                    chosen = rngs[ti].sample(cands, min(N, len(cands)))
                else:
                    chosen = sorted(cands, key=lambda c: (-c[0], c[1], c[2]))[:N]
                for _, nid, idx in chosen:
                    forked[ti].add((nid, idx))
                    reqs += [Request(ti, nid, idx)] * T
        if not reqs:
            break
        expand(trees, reqs, gen, max_response, max_model_len, tag, it)


# ---------------------------------------------------------------- P1
P1_SCORES = ("mu", "sigma", "mu_sigma", "random")


def p1_candidates(seg: SegTree, score: str, lam: float, min_suffix: int,
                  root_eligible: bool = True) -> List[Tuple[SegNode, float]]:
    """Eligible states with S_explore. Reward-free by construction (proposal §4.2)."""
    out = []
    for v in seg.nodes:
        if v.is_root and not root_eligible:
            continue
        if not seg.before_boxed(v) or seg.mean_suffix_len(v) < min_suffix:
            continue
        mu, sd, _ = seg.ce_stats(v)
        if mu is None:
            continue
        s = {"mu": mu, "sigma": sd, "mu_sigma": mu + lam * sd, "random": 0.0}[score]
        out.append((v, s))
    return out


def run_p1(trees, gen, rngs, budget, max_response, max_model_len, score="mu_sigma", lam=1.0,
           per_round=1, min_suffix=32, root_eligible=True):
    """Build T0 with exactly `budget` trajectories per problem.

    per_round=1 is Algorithm 1; per_round>1 is the batched-branching ablation.
    """
    assert score in P1_SCORES, score
    expand(trees, [Request(i, None, 0) for i in range(len(trees))], gen, max_response,
           max_model_len, "p1", 0)
    rnd = 0
    while True:
        rnd += 1
        reqs = []
        for ti, t in enumerate(trees):
            need = budget - len(t.nodes)
            if need <= 0:
                continue
            seg = SegTree(t)
            cands = p1_candidates(seg, score, lam, min_suffix, root_eligible)
            rngs[ti].shuffle(cands)  # random tie-breaking
            b = min(per_round, need)
            if score == "random":
                chosen = [v for v, _ in cands[:b]]
            else:
                chosen = [v for v, _ in sorted(cands, key=lambda c: -c[1])[:b]]
            if not chosen:  # nothing eligible: fall back to a fresh chain
                reqs += [Request(ti, None, 0)] * b
                continue
            for v in chosen:
                parent, idx = seg.fork_point(v)
                reqs.append(Request(ti, parent, idx))
        if not reqs:
            break
        before = sum(len(t.nodes) for t in trees)
        expand(trees, reqs, gen, max_response, max_model_len, "p1", rnd)
        if sum(len(t.nodes) for t in trees) == before:  # every request was out of room
            break


# ---------------------------------------------------------------- P2
P2_VARIANTS = ("rand", "unif", "delta", "absdelta", "spread")


def _softmax(xs: Sequence[float], tau: float) -> List[float]:
    m = max(xs)
    ws = [math.exp((x - m) / tau) for x in xs]
    z = sum(ws)
    return [w / z for w in ws]


def state_distribution(seg: SegTree, variant: str, tau: float = 0.5,
                       candidates: str = "all") -> Dict[int, float]:
    """Allocation over expansion *states* u = parent(v), keyed by SegNode.idx.

    `candidates`: "all" non-root nodes (proposal eq. 11), or "branch_children" (only
    children of states with >=2 children -- the only nodes where delta can be non-zero).
    "unif" and "spread" score branch points directly; if a tree has none they fall back
    to "rand".
    """
    assert variant in P2_VARIANTS, variant
    if variant in ("unif", "spread"):
        bps = [u for u in seg.branch_points() if seg.before_boxed(u)]
        if bps:
            if variant == "unif":
                return {u.idx: 1.0 / len(bps) for u in bps}
            spreads = [statistics.pstdev([seg.value(c) for c in u.children]) for u in bps]
            return {u.idx: p for u, p in zip(bps, _softmax(spreads, tau))}
        variant = "rand"

    vs = [v for v in seg.nodes if not v.is_root and seg.before_boxed(v.parent)]
    if candidates == "branch_children":
        vs = [v for v in vs if len(v.parent.children) >= 2] or vs
    if not vs:
        return {seg.root.idx: 1.0}
    if variant == "rand":
        probs = [1.0 / len(vs)] * len(vs)
    else:
        ds = [seg.delta(v) for v in vs]
        probs = _softmax([abs(d) for d in ds] if variant == "absdelta" else ds, tau)
    dist: Dict[int, float] = {}
    for v, p in zip(vs, probs):
        dist[v.parent.idx] = dist.get(v.parent.idx, 0.0) + p
    return dist


def run_p2(trees, gen, verifier, rngs, k, variant, max_response, max_model_len, tau=0.5,
           candidates="all"):
    """Spend the reserved k rollouts. Trees must be scored P1 trees. p is computed once
    per tree (Algorithm 1 line 15), so all k rollouts go out in one batched call."""
    reqs = []
    for ti, t in enumerate(trees):
        seg = SegTree(t)
        dist = state_distribution(seg, variant, tau, candidates)
        states = list(dist)
        picks = rngs[ti].choices(states, weights=[dist[s] for s in states], k=k)
        t.meta["p2"] = {
            "variant": variant, "tau": tau, "k": k, "candidates": candidates,
            "dist": {"%d:%d" % seg.state_key(seg.nodes[s]): p for s, p in dist.items()},
        }
        for s in picks:
            parent, idx = seg.fork_point(seg.nodes[s])
            reqs.append(Request(ti, parent, idx))
    expand(trees, reqs, gen, max_response, max_model_len, "p2", 0)
    score(trees, gen, verifier)
