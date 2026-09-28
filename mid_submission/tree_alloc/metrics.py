"""Per-tree metrics computed offline from JSONL logs, plus small stats helpers
(paired bootstrap) so the analysis needs no scipy."""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Sequence

from .segments import SegTree
from .tree import Tree


def tree_metrics(t: Tree) -> Dict:
    rewards = [n.reward for n in t.nodes]
    scored = all(r is not None for r in rewards)
    seg = SegTree(t)
    nonroot = [v for v in seg.nodes if not v.is_root]
    m = {
        "problem_id": t.problem_id,
        "n_leaves": len(t.nodes),
        "new_tokens": t.new_tokens,
        "n_seg_nodes": len(nonroot),
        "n_branch_points": len(seg.branch_points()),
        "truncated_frac": sum(n.finish_reason == "length" for n in t.nodes) / len(t.nodes),
        "distinct_answers": len({n.answer for n in t.nodes if n.answer is not None}),
        # Fig. 8: relative position of each fork within the parent trajectory
        "fork_rel_pos": [n.offset / t.nodes[n.parent].end for n in t.nodes if n.parent is not None],
        # Fig. 7: token id that was replaced at each fork
        "fork_tokens": [t.nodes[n.parent].token_ids[n.fork_idx] for n in t.nodes
                        if n.parent is not None and n.fork_idx < len(t.nodes[n.parent].token_ids)],
    }
    if scored:
        acc = sum(rewards) / len(rewards)
        m.update({
            "acc": acc,
            "pass_any": float(max(rewards) > 0),
            "mixed": float(0 < acc < 1),
            # branch points whose children disagree in value (sibling outcomes differ)
            "mixed_branch_frac": (sum(len({seg.value(c) for c in u.children}) > 1
                                      for u in seg.branch_points()) / max(len(seg.branch_points()), 1)),
        })
    return m


def mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def paired_bootstrap(a: Dict[str, float], b: Dict[str, float], n: int = 2000, seed: int = 0,
                     stat: Callable[[List[float]], float] = mean) -> Dict[str, float]:
    """CI of mean(a - b) over problems present in both (resampling problems)."""
    keys = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in keys if not math.isnan(a[k]) and not math.isnan(b[k])]
    if not diffs:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(seed)
    boots = sorted(stat([rng.choice(diffs) for _ in diffs]) for _ in range(n))
    return {"diff": stat(diffs), "lo": boots[int(0.025 * n)], "hi": boots[int(0.975 * n) - 1],
            "p_le_0": sum(x <= 0 for x in boots) / n, "n": len(diffs)}
