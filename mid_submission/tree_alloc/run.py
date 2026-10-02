"""JSONL helpers and a per-log summary table.

  python -m tree_alloc.run logs/task2/*.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, Iterable, List

from .metrics import mean, tree_metrics
from .tree import Tree


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


def summary(paths: List[str]) -> None:
    cols = ["n_leaves", "new_tokens", "acc", "pass_any", "mixed", "distinct_answers", "truncated_frac"]
    print("| log | problems | " + " | ".join(cols) + " | fork_rel_pos |")
    print("|" + "---|" * (len(cols) + 3))
    for path in paths:
        ms = [tree_metrics(Tree.from_json(r)) for r in read_jsonl(path)]
        if not ms:
            continue
        row = [f"{mean([m.get(c) for m in ms]):.3g}" for c in cols]
        pos = [p for m in ms for p in m["fork_rel_pos"]]
        print(f"| {os.path.basename(path)} | {len(ms)} | " + " | ".join(row) + f" | {mean(pos):.2f} |")


if __name__ == "__main__":
    summary(sys.argv[1:])
