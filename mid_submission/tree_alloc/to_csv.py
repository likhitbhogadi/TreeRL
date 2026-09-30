"""JSONL tree logs -> CSV (one row per tree, one row per node), plus results.md tables -> CSV.

  python -m tree_alloc.to_csv --logs logs --out results/csv [--results results/task2/results.md]

Every *.jsonl under --logs (recursively) is streamed, so the 700 MB logs never sit in memory.
Per-token logprobs are not exported (~50M rows); the JSONL logs stay the source of truth.
  trees.csv  log, method, M, N, L, T, problem_id, gold, n_leaves, new_tokens, acc, pass_any, mixed,
             distinct_answers, truncated_frac, n_forks, mean_fork_rel_pos
  nodes.csv  one row per node = one leaf trajectory: tree position, lengths, reward, answer,
             TreeRL's own grade, and for forked nodes the replaced token, its surprisal and position
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

from .tree import Tree
from .verify import normalize

TREE_COLS = ["log", "method", "M", "N", "L", "T", "problem_id", "gold", "n_leaves", "new_tokens", "acc",
             "pass_any", "mixed", "distinct_answers", "truncated_frac", "n_forks", "mean_fork_rel_pos"]
NODE_COLS = ["log", "method", "M", "N", "L", "T", "problem_id", "node_id", "parent", "depth", "fork_idx",
             "offset", "new_tokens", "response_len", "finish_reason", "reward", "treerl_grade", "answer",
             "n_segments", "boxed_at", "mean_logprob", "fork_token_id", "fork_token", "fork_surprisal",
             "fork_rel_pos"]


def rows(path, log, decode):
    """(tree_row, [node_rows]) per line of one JSONL log."""
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            t = Tree.from_json(d)
            cfg = d.get("meta", {}).get("cfg", {})
            base = {"log": log, "method": d.get("meta", {}).get("method"),
                    **{k: cfg.get(k) for k in ("M", "N", "L", "T")}, "problem_id": t.problem_id}
            grades = d.get("meta", {}).get("treerl_grade") or []
            depth, nodes, pos = {}, [], []
            for n in t.nodes:
                depth[n.id] = 0 if n.parent is None else depth[n.parent] + 1
                r = {**base, "node_id": n.id, "parent": n.parent, "depth": depth[n.id], "fork_idx": n.fork_idx,
                     "offset": n.offset, "new_tokens": len(n.token_ids), "response_len": n.end,
                     "finish_reason": n.finish_reason, "reward": n.reward,
                     "treerl_grade": grades[n.id] if n.id < len(grades) else None, "answer": n.answer,
                     "n_segments": len(n.seg_ends), "boxed_at": n.boxed_at,
                     "mean_logprob": sum(n.logprobs) / len(n.logprobs) if n.logprobs else None}
                if n.parent is not None:
                    par = t.nodes[n.parent]
                    r["fork_rel_pos"] = n.offset / par.end
                    pos.append(r["fork_rel_pos"])
                    if n.fork_idx < len(par.token_ids):
                        r["fork_token_id"] = par.token_ids[n.fork_idx]
                        r["fork_token"] = decode(r["fork_token_id"])
                        r["fork_surprisal"] = -par.logprobs[n.fork_idx]
                nodes.append(r)
            rs = [n.reward for n in t.nodes]
            tree = {**base, "gold": t.gold, "n_leaves": len(t.nodes), "new_tokens": t.new_tokens,
                    "distinct_answers": len({normalize(n.answer) for n in t.nodes if n.answer is not None}),
                    "truncated_frac": sum(n.finish_reason == "length" for n in t.nodes) / len(t.nodes),
                    "n_forks": len(pos), "mean_fork_rel_pos": sum(pos) / len(pos) if pos else None}
            if all(r is not None for r in rs):
                acc = sum(rs) / len(rs)
                tree.update(acc=acc, pass_any=float(max(rs) > 0), mixed=float(0 < acc < 1))
            yield tree, nodes


def md_tables(md_path, out):
    """Each '## heading' + markdown table in results.md -> <out>/<heading slug>.csv."""
    title, table, written = None, [], []

    def flush():
        if title and table:
            path = os.path.join(out, re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + ".csv")
            with open(path, "w", newline="") as f:
                csv.writer(f).writerows(r for r in table if not all(set(c) <= set("-: ") for c in r))
            written.append(path)

    for line in open(md_path):
        if line.startswith("## "):
            flush()
            title, table = line[3:].strip(), []
        elif line.startswith("|"):
            table.append([c.strip().strip("`") for c in line.strip().strip("|").split("|")])
    flush()
    return written


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", nargs="*", default=[], help="results.md files to convert too")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    cache = {}
    decode = lambda t: cache.setdefault(t, tok.decode([t]))  # noqa: E731

    paths = sorted(glob.glob(os.path.join(a.logs, "**", "*.jsonl"), recursive=True))
    with open(os.path.join(a.out, "trees.csv"), "w", newline="") as ft, \
            open(os.path.join(a.out, "nodes.csv"), "w", newline="") as fn:
        wt = csv.DictWriter(ft, TREE_COLS)
        wn = csv.DictWriter(fn, NODE_COLS)
        wt.writeheader()
        wn.writeheader()
        for path in paths:
            log = os.path.relpath(path, a.logs)[: -len(".jsonl")]
            n = 0
            for tree, nodes in rows(path, log, decode):
                wt.writerow(tree)
                wn.writerows(nodes)
                n += 1
            print(f"{log}: {n} trees", flush=True)
    for md in a.results:
        for p in md_tables(md, a.out):
            print("wrote", p)


if __name__ == "__main__":
    main()
