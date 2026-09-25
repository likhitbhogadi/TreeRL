"""Segment-level view of a rollout tree (proposal §4.1): a node is a "\\n\\n"-delimited
reasoning segment, and trajectories sharing a prefix share segment nodes.

Built as a trie over each leaf's segment token-tuples, so it works for any tree,
including EPTree trees whose forks fall mid-segment (the diverging segment simply
becomes a sibling).

Quantities (all from logged logprobs, no extra forward passes):
  V(v)      mean leaf reward over L(v)                          (proposal eq. 2/9)
  delta(v)  V(v) - V(parent(v))                                 (eq. 10)
  H_{l,v}   mean surprisal of leaf l's tokens after v            (eq. 4)
  mu_v, sigma_v  mean / population std of H_{l,v} over L(v)     (eqs. 6-7)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .tree import Tree


class SegNode:
    __slots__ = ("idx", "parent", "depth", "end_pos", "children", "leaves", "ends", "_by_key")

    def __init__(self, idx: int, parent: Optional["SegNode"], end_pos: int):
        self.idx = idx
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.end_pos = end_pos  # response length of the state *after* this segment
        self.children: List[SegNode] = []
        self.leaves: List[int] = []  # node ids of trajectories passing through or ending here
        self.ends: List[int] = []  # node ids of trajectories ending exactly here
        self._by_key: Dict[Tuple[int, ...], SegNode] = {}

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def __repr__(self) -> str:
        return f"SegNode(idx={self.idx}, depth={self.depth}, end={self.end_pos}, leaves={len(self.leaves)})"


class SegTree:
    def __init__(self, tree: Tree):
        self.tree = tree
        self.root = SegNode(0, None, 0)
        self.nodes: List[SegNode] = [self.root]
        self._suffix: Dict[int, List[float]] = {}  # leaf -> suffix sums of surprisal
        self._len: Dict[int, int] = {}
        self._boxed: Dict[int, Optional[int]] = {}
        for n in tree.nodes:
            self._insert(n.id)

    def _insert(self, leaf: int) -> None:
        ids, lps = self.tree.response(leaf)
        suf = [0.0] * (len(lps) + 1)
        for i in range(len(lps) - 1, -1, -1):
            suf[i] = suf[i + 1] - lps[i]
        self._suffix[leaf] = suf
        self._len[leaf] = len(ids)
        self._boxed[leaf] = self.tree.boxed_pos(leaf)

        cur = self.root
        cur.leaves.append(leaf)
        start = 0
        for b in self.tree.boundaries(leaf):
            key = tuple(ids[start:b])
            nxt = cur._by_key.get(key)
            if nxt is None:
                nxt = SegNode(len(self.nodes), cur, b)
                cur._by_key[key] = nxt
                cur.children.append(nxt)
                self.nodes.append(nxt)
            nxt.leaves.append(leaf)
            cur, start = nxt, b
        cur.ends.append(leaf)

    # ---- outcome statistics ------------------------------------------
    def reward(self, leaf: int) -> float:
        r = self.tree.nodes[leaf].reward
        if r is None:
            raise ValueError(f"leaf {leaf} has no reward; run scoring first")
        return r

    def value(self, v: SegNode) -> float:
        return sum(self.reward(l) for l in v.leaves) / len(v.leaves)

    def delta(self, v: SegNode) -> float:
        assert v.parent is not None
        return self.value(v) - self.value(v.parent)

    def branch_points(self) -> List[SegNode]:
        return [v for v in self.nodes if len(v.children) >= 2]

    # ---- continuation cross-entropy -----------------------------------
    def suffix_len(self, leaf: int, v: SegNode) -> int:
        return self._len[leaf] - v.end_pos

    def suffix_ce(self, leaf: int, v: SegNode) -> Optional[float]:
        n = self.suffix_len(leaf, v)
        if n <= 0:
            return None
        return self._suffix[leaf][v.end_pos] / n

    def ce_stats(self, v: SegNode) -> Tuple[Optional[float], float, int]:
        """(mu_v, sigma_v, #leaves with a non-empty continuation)."""
        hs = [h for h in (self.suffix_ce(l, v) for l in v.leaves) if h is not None]
        if not hs:
            return None, 0.0, 0
        mu = sum(hs) / len(hs)
        var = sum((h - mu) ** 2 for h in hs) / len(hs)
        return mu, math.sqrt(var), len(hs)

    def mean_suffix_len(self, v: SegNode) -> float:
        return sum(max(self.suffix_len(l, v), 0) for l in v.leaves) / len(v.leaves)

    def before_boxed(self, v: SegNode) -> bool:
        """State v's prefix contains no part of the final "\\boxed" (shared prefix, so any leaf works)."""
        b = self._boxed[v.leaves[0]]
        return b is None or v.end_pos <= b

    # ---- forking ------------------------------------------------------
    def fork_point(self, v: SegNode) -> Tuple[Optional[int], int]:
        """(parent node id, fork_idx) to generate a new continuation from state v."""
        return self.tree.locate(min(v.leaves), v.end_pos)

    def state_key(self, v: SegNode) -> Tuple[int, int]:
        """Stable id of the state after v: (smallest leaf through it, response position)."""
        return min(v.leaves), v.end_pos


def treerl_step_rewards(seg: SegTree) -> Dict[int, float]:
    """TreeRL step reward (G_A + L_A)/sqrt(|L(s)|), keyed by SegNode.idx (paper eq.;
    the TreeRL code instead RLOO-normalizes leaves and applies sqrt before differencing)."""
    v_root = seg.value(seg.root)
    out = {}
    for v in seg.nodes:
        if v.is_root:
            continue
        val = seg.value(v)
        g = val - v_root
        l = val - seg.value(v.parent)
        out[v.idx] = (g + l) / math.sqrt(len(v.leaves))
    return out
