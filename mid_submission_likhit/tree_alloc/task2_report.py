"""Task 2 report: EPTree replication (B0 / B1 / B2) from JSONL logs.

  python -m tree_alloc.task2_report --logs logs/task2 --out results/task2

Expects b0_<n>.jsonl (i.i.d., n chains) and b{1,2}_<M>-<N>-<L>-<T>.jsonl. Writes
results.md (tables) and fig4/fig5/fig7/fig8 PNGs. All comparisons use the problems
present in every log. The i.i.d. curve is computed from the single b0 run with the
unbiased pass@k estimator (and first-k chains for tokens / distinct answers).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
from collections import Counter
from typing import Dict, List

from .metrics import paired_bootstrap
from .run import read_jsonl
from .tree import Tree
from .verify import normalize

# Reference palette (dataviz skill), light mode. Slots 1-3 validate all-pairs.
C_EPTREE, C_RANDOM, C_IID = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(path: str) -> Dict[str, Tree]:
    return {r["problem_id"]: Tree.from_json(r) for r in read_jsonl(path)}


def distinct(nodes) -> int:
    return len({normalize(n.answer) for n in nodes if n.answer is not None})


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def tree_stats(trees: Dict[str, Tree], pids: List[str]) -> Dict[str, Dict[str, float]]:
    out = {"pass": {}, "acc": {}, "tokens": {}, "leaves": {}, "distinct": {}, "mixed": {}}
    for p in pids:
        t = trees[p]
        rs = [n.reward for n in t.nodes]
        out["pass"][p] = float(max(rs) > 0)
        out["acc"][p] = sum(rs) / len(rs)
        out["tokens"][p] = t.new_tokens
        out["leaves"][p] = len(t.nodes)
        out["distinct"][p] = distinct(t.nodes)
        out["mixed"][p] = float(0 < sum(rs) < len(rs))
    return out


def mean(d: Dict[str, float]) -> float:
    return sum(d.values()) / len(d)


def iid_curve(b0: Dict[str, Tree], pids: List[str], ks: List[int]):
    rows = []
    for k in ks:
        passk, toks, dist = [], [], []
        for p in pids:
            nodes = b0[p].nodes
            c = sum(n.reward > 0 for n in nodes)
            passk.append(pass_at_k(len(nodes), c, k))
            toks.append(sum(len(n.token_ids) for n in nodes[:k]))
            dist.append(distinct(nodes[:k]))
        rows.append({"k": k, "pass": sum(passk) / len(pids), "tokens": sum(toks) / len(pids),
                     "distinct": sum(dist) / len(pids)})
    return rows


def fork_info(trees: Dict[str, Tree], pids: List[str]):
    pos, toks, surprisal = [], Counter(), []
    for p in pids:
        t = trees[p]
        for n in t.nodes:
            if n.parent is None:
                continue
            par = t.nodes[n.parent]
            pos.append(n.offset / par.end)
            toks[par.token_ids[n.fork_idx]] += 1
            surprisal.append(-par.logprobs[n.fork_idx])
    return pos, toks, surprisal


def show_token(text: str) -> str:
    """Visible token text: spaces as ␣, newlines/tabs as escapes, backslashes as-is."""
    return text.replace(" ", "␣").replace("\n", "\\n").replace("\t", "\\t") or "∅"


# ---------------------------------------------------------------- plotting
def _style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def _log_axis(ax, which="x", base=10):
    from matplotlib.ticker import FuncFormatter, NullFormatter

    fmt = FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log", base=base)
    axis.set_major_formatter(fmt)
    axis.set_minor_formatter(NullFormatter())


def _fig(plt, ncols=1, w=6.4, h=4.2):
    fig, axes = plt.subplots(1, ncols, figsize=(w * ncols, h), facecolor=SURFACE)
    return fig, (axes if ncols > 1 else [axes])


def _scatter_series(ax, pts, color, label, annotate):
    """Plot points; return (x, y, name) labels for `place_labels` when annotate is set."""
    ax.scatter([x for x, _, _ in pts], [y for _, y, _ in pts], s=64, color=color, label=label,
               edgecolors=SURFACE, linewidths=2, zorder=3)
    return [p for p in pts if p[2]] if annotate else []


def place_labels(ax, labels, obstacles, fontsize=8):
    """Greedy label placement: for each label try 8 offsets around its point and keep the one
    overlapping the fewest markers / already-placed labels (all in display points). Call after
    the axes' limits and the figure layout are final."""
    to_pt = 72.0 / ax.figure.dpi
    disp = lambda x, y: tuple(v * to_pt for v in ax.transData.transform((x, y)))  # noqa: E731
    marks = [disp(x, y) for x, y in obstacles]
    (bx0, by0), (bx1, by1) = [(u * to_pt, v * to_pt) for u, v in ax.get_window_extent().get_points()]
    placed = []
    cands = [(8, -4, "left"), (-8, -4, "right"), (0, 9, "center"), (0, -16, "center"),
             (8, 8, "left"), (-8, 8, "right"), (8, -16, "left"), (-8, -16, "right")]
    for x, y, name in sorted(labels, key=lambda l: -l[1]):
        px, py = disp(x, y)
        w, h = 0.62 * fontsize * len(name), fontsize + 2
        best = None
        for dx, dy, ha in cands:
            x0 = px + dx - (w if ha == "right" else w / 2 if ha == "center" else 0)
            rect = (x0, py + dy, x0 + w, py + dy + h)
            cost = sum(rect[0] - 6 < mx < rect[2] + 6 and rect[1] - 6 < my < rect[3] + 6 for mx, my in marks)
            cost += sum(not (rect[2] < r[0] or rect[0] > r[2] or rect[3] < r[1] or rect[1] > r[3]) for r in placed)
            cost += 3 * (rect[0] < bx0 or rect[2] > bx1 or rect[1] < by0 or rect[3] > by1)  # stay inside
            if best is None or cost < best[0]:
                best = (cost, dx, dy, ha, rect)
        _, dx, dy, ha, rect = best
        placed.append(rect)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=fontsize,
                    color=INK2, zorder=4)


def plot_all(out, iid, b2_rows, b1_rows, fork_b2, fork_b1, tok_decode):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 5: PassRate vs generated tokens -- full range, and a zoom on the tree budgets
    fig, axes = _fig(plt, ncols=2, w=5.4)
    panel_labels = []
    lo = min(r["tokens"] for r in b2_rows + b1_rows) * 0.85
    for ax, zoom in ((axes[0], False), (axes[1], True)):
        pts = [r for r in iid if not zoom or r["tokens"] >= lo * 0.7]
        ax.plot([r["tokens"] for r in pts], [100 * r["pass"] for r in pts], color=C_IID, linewidth=2,
                marker="o", markersize=5, label="i.i.d. chains (pass@k)", zorder=2)
        k_labels = [(r["tokens"], 100 * r["pass"], f"k={r['k']}") for r in pts
                    if r["k"] in ((8, 16, 32, 64) if zoom else (1, 4))]
        labels = _scatter_series(ax, [(r["tokens"], 100 * r["pass"], r["cfg"]) for r in b2_rows], C_EPTREE,
                                 "EPTree (B2)", zoom)
        _scatter_series(ax, [(r["tokens"], 100 * r["pass"], "") for r in b1_rows], C_RANDOM,
                        "Random forks (B1)", False)
        panel_labels.append((ax, labels + k_labels,
                             [(r["tokens"], 100 * r["pass"]) for r in b2_rows + b1_rows + pts]))
        _log_axis(ax)
        _style(ax, "Generated tokens per problem (log scale)", "PassRate (% problems with ≥1 correct leaf)")
        ax.set_title("Zoom: tree budgets (16-64 leaves)" if zoom else "PassRate vs. generation budget",
                     color=INK, fontsize=12, loc="left")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    for args in panel_labels:
        place_labels(*args)
    fig.savefig(os.path.join(out, "fig5_passrate_vs_tokens.png"), dpi=160)
    plt.close(fig)

    # Fig 4: responses and distinct answers vs tokens (two panels, one axis each)
    fig, axes = _fig(plt, ncols=2, w=5.2)
    for ax, key, ylabel, title in (
        (axes[0], "leaves", "Responses (leaves) per problem", "Responses vs. budget"),
        (axes[1], "distinct", "Distinct final answers per problem", "Answer diversity vs. budget"),
    ):
        ax.plot([r["tokens"] for r in iid], [r["k"] if key == "leaves" else r["distinct"] for r in iid],
                color=C_IID, linewidth=2, marker="o", markersize=5, label="i.i.d. chains", zorder=2)
        _scatter_series(ax, [(r["tokens"], r[key], "") for r in b2_rows], C_EPTREE, "EPTree (B2)", False)
        _scatter_series(ax, [(r["tokens"], r[key], "") for r in b1_rows], C_RANDOM, "Random forks (B1)", False)
        _log_axis(ax)
        if key == "leaves":
            _log_axis(ax, "y", base=2)
        _style(ax, "Generated tokens per problem (log scale)", ylabel)
        ax.set_title(title, color=INK, fontsize=12, loc="left")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_responses_diversity_vs_tokens.png"), dpi=160)
    plt.close(fig)

    # Fig 7: top-10 forking tokens (EPTree), one series
    top = fork_b2[1].most_common(10)
    total = sum(fork_b2[1].values())
    fig, (ax,) = _fig(plt, w=6.4, h=4.0)
    labels = [show_token(tok_decode(t)) for t, _ in top][::-1]
    vals = [100 * c / total for _, c in top][::-1]
    ax.barh(range(len(vals)), vals, color=C_EPTREE, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9, color=INK, family="monospace")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8, color=INK2)
    _style(ax, "Share of EPTree forks (%)", "")
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    ax.set_title("Top-10 tokens EPTree forks at", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig7_top_fork_tokens.png"), dpi=160)
    plt.close(fig)

    # Fig 8: relative fork position (frequency polygons, two series)
    bins = 20
    fig, (ax,) = _fig(plt)
    for (pos, _, _), color, name in ((fork_b2, C_EPTREE, "EPTree (B2)"), (fork_b1, C_RANDOM, "Random forks (B1)")):
        if not pos:
            continue
        counts = [0] * bins
        for p in pos:
            counts[min(int(p * bins), bins - 1)] += 1
        xs = [(i + 0.5) / bins for i in range(bins)]
        ys = [100 * c / len(pos) for c in counts]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4, label=name)
        i = 1 if color == C_EPTREE else 10  # EPTree label at its early peak, random mid-response
        ax.annotate(name, (xs[i], ys[i]), textcoords="offset points", xytext=(6, 8), fontsize=8, color=INK2)
    _style(ax, "Fork position / parent response length", "Share of forks (%)")
    ax.set_title("Where forks happen in the response", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig8_fork_position.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--no_plots", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    b0_path = sorted(glob.glob(os.path.join(a.logs, "b0_*.jsonl")))[-1]
    runs = {}
    for path in sorted(glob.glob(os.path.join(a.logs, "b[12]_*.jsonl"))):
        m = re.match(r"(b[12])_(\d+)-(\d+)-(\d+)-(\d+)\.jsonl", os.path.basename(path))
        if m:
            runs[(m.group(1), tuple(int(x) for x in m.groups()[1:]))] = load(path)
    b0 = load(b0_path)
    pids = sorted(set(b0).intersection(*[set(t) for t in runs.values()]))
    n0 = len(next(iter(b0.values())).nodes)
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= n0]
    iid = iid_curve(b0, pids, ks)

    rows = {}
    for (kind, cfg), trees in runs.items():
        s = tree_stats(trees, pids)
        M, N, L, T = cfg
        rows[(kind, cfg)] = {"cfg": f"({M},{N},{L},{T})", "stats": s,
                             **{k: mean(v) for k, v in s.items()}}
    b2_rows = [r for (k, _), r in sorted(rows.items()) if k == "b2"]
    b1_rows = [r for (k, _), r in sorted(rows.items()) if k == "b1"]

    def iid_at(tokens):  # pass@k interpolated at a token budget (log-linear)
        pts = [(r["tokens"], r["pass"]) for r in iid]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= tokens <= x1:
                f = (math.log(tokens) - math.log(x0)) / (math.log(x1) - math.log(x0))
                return y0 + f * (y1 - y0)
        return float("nan")

    L = []
    L.append(f"# Task 2 results: EPTree replication\n")
    L.append(f"{len(pids)} problems common to all runs; i.i.d. baseline from `{os.path.basename(b0_path)}` "
             f"({n0} chains/problem).\n")
    L.append("## i.i.d. chains (B0), unbiased pass@k\n")
    L.append("| k | tokens/problem | PassRate | distinct answers |\n|---|---|---|---|")
    for r in iid:
        L.append(f"| {r['k']} | {r['tokens']:.0f} | {100 * r['pass']:.1f}% | {r['distinct']:.2f} |")

    L.append("\n## Tree configs (Fig. 4 / Fig. 5 data)\n")
    L.append("| method | (M,N,L,T) | leaves | tokens/problem | PassRate | i.i.d. PassRate at same tokens | "
             "mean acc | distinct answers | mixed-outcome trees |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for (kind, cfg), r in sorted(rows.items(), key=lambda kv: (kv[1]["tokens"], kv[0][0])):
        name = "EPTree (B2)" if kind == "b2" else "Random (B1)"
        L.append(f"| {name} | {r['cfg']} | {r['leaves']:.0f} | {r['tokens']:.0f} | {100 * r['pass']:.1f}% | "
                 f"{100 * iid_at(r['tokens']):.1f}% | {r['acc']:.3f} | {r['distinct']:.2f} | {100 * r['mixed']:.0f}% |")

    L.append("\n## Table 2: entropy-guided vs. random forking (same M,N,L,T)\n")
    L.append("| (M,N,L,T) | PassRate B2 | PassRate B1 | Δ PassRate [95% CI] | acc B2 | acc B1 | "
             "distinct B2 | distinct B1 | tokens B2 | tokens B1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for (kind, cfg), r2 in sorted(rows.items()):
        if kind != "b2" or ("b1", cfg) not in rows:
            continue
        r1 = rows[("b1", cfg)]
        b = paired_bootstrap(r2["stats"]["pass"], r1["stats"]["pass"])
        L.append(f"| {r2['cfg']} | {100 * r2['pass']:.1f}% | {100 * r1['pass']:.1f}% | "
                 f"{100 * b['diff']:+.1f} [{100 * b['lo']:+.1f}, {100 * b['hi']:+.1f}] | {r2['acc']:.3f} | "
                 f"{r1['acc']:.3f} | {r2['distinct']:.2f} | {r1['distinct']:.2f} | {r2['tokens']:.0f} | {r1['tokens']:.0f} |")

    # pool forks over configs
    fork_b2 = ([], Counter(), [])
    fork_b1 = ([], Counter(), [])
    for (kind, _), trees in runs.items():
        pos, toks, sur = fork_info(trees, pids)
        tgt = fork_b2 if kind == "b2" else fork_b1
        tgt[0].extend(pos)
        tgt[1].update(toks)
        tgt[2].extend(sur)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    decode = lambda t: tok.decode([t])  # noqa: E731
    L.append("\n## Fig. 7: top-10 forking tokens (EPTree, all configs pooled)\n")
    L.append("| rank | token | share of forks |\n|---|---|---|")
    tot = sum(fork_b2[1].values())
    for i, (t, c) in enumerate(fork_b2[1].most_common(10), 1):
        L.append(f"| {i} | `{show_token(decode(t))}` | {100 * c / tot:.1f}% |")
    L.append("\n## Fig. 8: fork positions and surprisal\n")
    L.append("| method | forks | mean relative position | median relative position | "
             "mean surprisal at fork token |\n|---|---|---|---|---|")
    for name, (pos, _, sur) in (("EPTree (B2)", fork_b2), ("Random (B1)", fork_b1)):
        if pos:
            sp = sorted(pos)
            L.append(f"| {name} | {len(pos)} | {sum(pos) / len(pos):.2f} | {sp[len(sp) // 2]:.2f} | "
                     f"{sum(sur) / len(sur):.2f} |")

    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    if not a.no_plots:
        plot_all(a.out, iid, b2_rows, b1_rows, fork_b2, fork_b1, decode)
        print(f"\nfigures written to {a.out}")


if __name__ == "__main__":
    main()
