"""Task 5 oracle: estimate each candidate expansion state's true outcome statistics.

For a state u (a prefix), with Q(u,a) the success rate after next step a:
    p(1-p) = Var_a[Q(u,a)] + E_a[Q(u,a)(1 - Q(u,a))]
`var_total` = p_hat(1-p_hat) from R full rollouts: outcome variance at u.
`var_between` = variance of Q_hat over K sampled next segments (M rollouts each):
    how much the *decision* at u matters -- the quantity the proposal's claim is about.

All P2 variants share the same P1 tree, so each variant is scored by its expected
oracle value sum_u p(u) * metric(u) -- no need to generate the k rollouts, and no
sampling noise from doing so.
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Sequence, Tuple

from .gen import Generator
from .methods import state_distribution
from .segments import SegNode, SegTree
from .tree import Tree
from .verify import Verifier


def candidate_states(seg: SegTree, variants: Sequence[Tuple[str, float, str]],
                     max_states: int) -> Tuple[List[SegNode], Dict[str, Dict[int, float]]]:
    """Union of the supports of every (variant, tau, candidates) distribution, capped at
    `max_states` by highest probability under any variant."""
    dists = {f"{v}|{tau}|{c}": state_distribution(seg, v, tau, c) for v, tau, c in variants}
    best: Dict[int, float] = {}
    for d in dists.values():
        for s, p in d.items():
            best[s] = max(best.get(s, 0.0), p)
    keep = sorted(best, key=lambda s: (-best[s], s))[:max_states]
    return [seg.nodes[s] for s in keep], dists


def _rewards(gen, verifier, tree_prefixes, gens) -> List[float]:
    out = []
    for (t, prefix), g in zip(tree_prefixes, gens):
        if g.finish_reason != "stop":
            out.append(0.0)
            continue
        out.append(verifier(gen.decode(prefix + g.token_ids), t.gold))
    return out


def run_oracle(trees: List[Tree], states: List[List[SegNode]], gen: Generator, verifier: Verifier,
               R: int, K: int, M: int, max_response: int, max_model_len: int) -> List[Dict]:
    segs_prefix = []  # (tree, state, prefix ids)
    for t, sts in zip(trees, states):
        seg = SegTree(t)
        for u in sts:
            parent, idx = seg.fork_point(u)
            segs_prefix.append((t, seg, u, t.prefix_ids(parent, idx)))

    def room(t, prefix):
        return min(max_response - len(prefix), max_model_len - len(t.prompt_ids) - len(prefix))

    # full rollouts -> p_hat
    meta, prompts, mts = [], [], []
    for si, (t, _, _, prefix) in enumerate(segs_prefix):
        if room(t, prefix) <= 0:
            continue
        for _ in range(R):
            meta.append((si, (t, prefix)))
            prompts.append(t.prompt_ids + prefix)
            mts.append(room(t, prefix))
    rewards = _rewards(gen, verifier, [m[1] for m in meta], gen.generate(prompts, mts)) if prompts else []
    full: Dict[int, List[float]] = {}
    for (si, _), r in zip(meta, rewards):
        full.setdefault(si, []).append(r)

    # next-step decisions -> Q_hat per action
    between: Dict[int, List[float]] = {}
    if K > 0 and M > 0:
        step_meta, prompts, mts = [], [], []
        for si, (t, _, _, prefix) in enumerate(segs_prefix):
            if room(t, prefix) <= 0:
                continue
            for _ in range(K):
                step_meta.append((si, prefix))
                prompts.append(t.prompt_ids + prefix)
                mts.append(room(t, prefix))
        steps = gen.generate(prompts, mts, stop=["\n\n"]) if prompts else []
        roll_meta, prompts, mts = [], [], []
        for a, ((si, prefix), s) in enumerate(zip(step_meta, steps)):
            t = segs_prefix[si][0]
            pre = prefix + s.token_ids
            if room(t, pre) <= 0 or s.finish_reason == "length":
                continue
            for _ in range(M):
                roll_meta.append((si, a, (t, pre)))
                prompts.append(t.prompt_ids + pre)
                mts.append(room(t, pre))
        rs = _rewards(gen, verifier, [m[2] for m in roll_meta], gen.generate(prompts, mts)) if prompts else []
        per_action: Dict[Tuple[int, int], List[float]] = {}
        for (si, a, _), r in zip(roll_meta, rs):
            per_action.setdefault((si, a), []).append(r)
        for (si, _), q in per_action.items():
            between.setdefault(si, []).append(sum(q) / len(q))

    records = []
    for si, (t, seg, u, _) in enumerate(segs_prefix):
        rs = full.get(si)
        if not rs:
            continue
        p = sum(rs) / len(rs)
        qs = between.get(si, [])
        records.append({
            "problem_id": t.problem_id,
            "state_key": "%d:%d" % seg.state_key(u),
            "depth": u.depth,
            "end_pos": u.end_pos,
            "n_children": len(u.children),
            "n_leaves_tree": len(u.leaves),
            "V_tree": seg.value(u),
            "p_hat": p,
            "var_total": p * (1 - p),
            "n_rollouts": len(rs),
            "q_hat": qs,
            "var_between": statistics.pvariance(qs) if len(qs) >= 2 else None,
        })
    return records


def expected_score(dist: Dict[str, float], by_key: Dict[str, Dict], metric: str) -> Tuple[float, float]:
    """(sum_u p(u) * metric(u) renormalized over states with an oracle value, covered mass)."""
    num = den = 0.0
    for key, p in dist.items():
        rec = by_key.get(key)
        if rec is None or rec.get(metric) is None:
            continue
        num += p * rec[metric]
        den += p
    return (num / den if den > 0 else float("nan")), den
