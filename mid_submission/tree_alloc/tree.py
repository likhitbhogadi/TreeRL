"""Rollout tree built from generation calls.

Layout follows TreeRL's TreeNode (openrlhf/trainer/ppo_utils/tree_node.py): every
generation call is one `Node` that continues its parent's response prefix
`parent.response[:parent.offset + fork_idx]`. The trajectory ending at a node is
one leaf, so #leaves == #nodes. This is the log format only: trees are built by TreeRL's
code (run_treerl.py converts its TreeNode lists into this structure).

Positions are always *response* positions (prompt excluded). "Forking at position
p" means the new generation is conditioned on response[:p] and produces a new
token at index p.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class Generation:
    token_ids: List[int]
    logprobs: List[float]
    finish_reason: str = "stop"  # "stop" | "length"


@dataclass
class Node:
    id: int
    parent: Optional[int]  # None: generated directly from the prompt
    fork_idx: int  # index into parent's token_ids where this node branches off
    offset: int  # response position of token_ids[0]
    token_ids: List[int]
    logprobs: List[float]
    finish_reason: str = "stop"
    seg_ends: List[int] = field(default_factory=list)  # local e: a segment ends after token e-1
    boxed_at: Optional[int] = None  # local index of the first token of "\boxed"
    reward: Optional[float] = None
    answer: Optional[str] = None
    phase: str = ""
    round: int = 0

    @property
    def end(self) -> int:
        return self.offset + len(self.token_ids)


def annotate(token_ids: List[int], token_text: Callable[[int], str]) -> Tuple[List[int], Optional[int]]:
    """Segment ends ("\\n\\n" boundaries) and the start of the first "\\boxed", from token ids.

    A boundary is placed after token i when the text so far ends with a blank line
    and the next token does not continue the newline run. Works whether "\\n\\n" is
    its own token, merged (".\\n\\n"), or split over two "\\n" tokens.
    """
    texts = [token_text(t) for t in token_ids]
    seg_ends: List[int] = []
    tail = ""
    for i, s in enumerate(texts):
        tail = (tail + s)[-8:]
        nxt = texts[i + 1] if i + 1 < len(texts) else ""
        if tail.rstrip(" \t").endswith("\n\n") and not nxt.startswith("\n"):
            seg_ends.append(i + 1)

    boxed_at = None
    window = ""
    for i, s in enumerate(texts):
        window = (window + s)[-32:]
        if "\\boxed" in window:
            j = i
            while "\\boxed" not in "".join(texts[j:i + 1]):
                j -= 1
            boxed_at = j
            break
    return seg_ends, boxed_at


class Tree:
    def __init__(self, problem_id: str, prompt_ids: List[int], gold: Optional[str] = None,
                 meta: Optional[Dict] = None):
        self.problem_id = problem_id
        self.prompt_ids = prompt_ids
        self.gold = gold
        self.meta = meta or {}
        self.nodes: List[Node] = []

    # ---- construction -------------------------------------------------
    def add(self, parent: Optional[int], fork_idx: int, gen: Generation,
            token_text: Callable[[int], str], phase: str = "", round: int = 0) -> Node:
        if parent is None:
            assert fork_idx == 0, "root generations fork at 0"
            offset = 0
        else:
            p = self.nodes[parent]
            assert 0 <= fork_idx <= len(p.token_ids), (fork_idx, len(p.token_ids))
            offset = p.offset + fork_idx
        seg_ends, boxed_at = annotate(gen.token_ids, token_text)
        node = Node(
            id=len(self.nodes), parent=parent, fork_idx=fork_idx, offset=offset,
            token_ids=list(gen.token_ids), logprobs=list(gen.logprobs),
            finish_reason=gen.finish_reason, seg_ends=seg_ends, boxed_at=boxed_at,
            phase=phase, round=round,
        )
        self.nodes.append(node)
        return node

    # ---- trajectory views ---------------------------------------------
    def path(self, node_id: int) -> List[Tuple[Node, int]]:
        """(node, local end) pairs from the prompt down to `node_id`'s leaf."""
        chain = []
        n = self.nodes[node_id]
        end = len(n.token_ids)
        while True:
            chain.append((n, end))
            if n.parent is None:
                break
            end = n.fork_idx
            n = self.nodes[n.parent]
        return chain[::-1]

    def response(self, node_id: int) -> Tuple[List[int], List[float]]:
        ids: List[int] = []
        lps: List[float] = []
        for n, end in self.path(node_id):
            ids += n.token_ids[:end]
            lps += n.logprobs[:end]
        return ids, lps

    def boundaries(self, node_id: int) -> List[int]:
        """Sorted segment end positions of the trajectory, always including its length."""
        out = set()
        for n, end in self.path(node_id):
            out.update(n.offset + e for e in n.seg_ends if e <= end)
        out.add(self.nodes[node_id].end)
        out.discard(0)
        return sorted(out)

    def boxed_pos(self, node_id: int) -> Optional[int]:
        for n, end in self.path(node_id):
            if n.boxed_at is not None and n.boxed_at < end:
                return n.offset + n.boxed_at
        return None

    def prefix_ids(self, parent: Optional[int], fork_idx: int) -> List[int]:
        if parent is None:
            return []
        ids, _ = self.response(parent)
        return ids[: self.nodes[parent].offset + fork_idx]

    @property
    def new_tokens(self) -> int:
        """Generated-token budget: shared prefixes are not re-counted."""
        return sum(len(n.token_ids) for n in self.nodes)

    # ---- serialization ------------------------------------------------
    def to_json(self) -> Dict:
        return {
            "problem_id": self.problem_id,
            "gold": self.gold,
            "meta": self.meta,
            "prompt_ids": self.prompt_ids,
            "new_tokens": self.new_tokens,
            "nodes": [asdict(n) for n in self.nodes],
        }

    @classmethod
    def from_json(cls, d: Dict) -> "Tree":
        t = cls(d["problem_id"], d["prompt_ids"], d.get("gold"), d.get("meta"))
        t.nodes = [Node(**n) for n in d["nodes"]]
        return t
