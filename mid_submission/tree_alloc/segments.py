"""Segment-level view of a rollout tree: a node is a "\\n\\n"-delimited reasoning segment, and
trajectories sharing a prefix share segment nodes.

Built as a trie over each leaf's segment token-tuples, so it works for any tree, including
EPTree trees whose forks fall mid-segment (the diverging segment simply becomes a sibling).
Used by the Task 2 analyses: V(v) = mean leaf reward below v, and branch points.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .tree import Tree


class SegNode:
    __slots__ = ("idx", "parent", "end_pos", "children", "leaves", "ends", "_by_key")

    def __init__(self, idx: int, parent: Optional["SegNode"], end_pos: int):
        self.idx = idx
        self.parent = parent
        self.end_pos = end_pos  # response length of the state *after* this segment
        self.children: List[SegNode] = []
        self.leaves: List[int] = []  # node ids of trajectories passing through or ending here
        self.ends: List[int] = []  # node ids of trajectories ending exactly here
        self._by_key: Dict[Tuple[int, ...], SegNode] = {}

    @property
    def is_root(self) -> bool:
        return self.parent is None


class SegTree:
    def __init__(self, tree: Tree):
        self.tree = tree
        self.root = SegNode(0, None, 0)
        self.nodes: List[SegNode] = [self.root]
        for n in tree.nodes:
            self._insert(n.id)

    def _insert(self, leaf: int) -> None:
        ids, _ = self.tree.response(leaf)
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

    def reward(self, leaf: int) -> float:
        r = self.tree.nodes[leaf].reward
        if r is None:
            raise ValueError(f"leaf {leaf} has no reward; run scoring first")
        return r

    def value(self, v: SegNode) -> float:
        return sum(self.reward(l) for l in v.leaves) / len(v.leaves)

    def branch_points(self) -> List[SegNode]:
        return [v for v in self.nodes if len(v.children) >= 2]
