"""Figures and numbers for the mid-submission report (report/), from data already in the repo.

  python -m tree_alloc.report_figures            # writes report/figures/*.pdf|png, prints the table numbers

Inputs: results/csv/trees.csv + nodes.csv (Task 2 trees, from to_csv.py), results/rl/per_problem/*.jsonl (RL evals).
Colours: validated categorical slots 1-3 (trees = blue, chains = aqua, third series = orange), each series also has
its own marker shape so the figures survive grayscale printing.
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os
import random
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .verify import normalize  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "report", "figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300,
    "pdf.fonttype": 42,
})
CONFIGS = ["2-1-1-1", "2-3-1-1", "4-1-1-1", "4-3-1-1", "6-2-1-2", "8-4-2-2"]


def boot_ci(vals, n=2000, seed=0):
    """mean and 95% bootstrap CI over the given per-problem values"""
    rng = random.Random(seed)
    m = sum(vals) / len(vals)
    bs = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return m, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def kfmt(v, _=None):
    return (f"{v / 1000:.1f}k".replace(".0k", "k") if v < 1e4 else f"{v / 1000:.0f}k") if v >= 1000 else f"{v:g}"


# ------------------------------------------------------------------------------------------------ Task 2
DIFF_BINS = [("never solved", 0, 0), ("sometimes", 1, 16), ("mostly solved", 17, 32)]  # solved x/32 by chains 33-64


def interp_w(tok_curve, ks, T):
    """(k_lo, k_hi, weight) to read a curve at token budget T, log-linear in tokens; None outside the range"""
    lo = max((k for k in ks if tok_curve[k] <= T), default=None)
    hi = min((k for k in ks if tok_curve[k] >= T), default=None)
    if lo is None or hi is None:
        return None
    w = 0 if lo == hi else (math.log(T) - math.log(tok_curve[lo])) / (math.log(tok_curve[hi]) - math.log(tok_curve[lo]))
    return lo, hi, w


def tree_markers(ax, rows, key, legend=False):
    """random forks as hollow squares underneath, EPTree as filled circles on top: both stay visible when a pair
    of runs has (near-)identical values, e.g. both (6,2,1,2) runs have PassRate 48.8"""
    for method, kw in (("b1", dict(marker="s", s=34, facecolors="none", edgecolors=ORANGE, linewidths=1.2, zorder=3,
                                    label="random forks")),
                       ("b2", dict(marker="o", s=16, color=BLUE, edgecolors="white", linewidths=0.4, zorder=4,
                                   label="EPTree"))):
        pts = sorted((r for r in rows if r["method"] == method), key=lambda r: r["tokens"])
        ax.plot([r["tokens"] for r in pts], [r[key] for r in pts], ls=":", lw=1.2, color=kw.get("edgecolors") if method == "b1"
                else BLUE, zorder=2)
        ax.scatter([r["tokens"] for r in pts], [r[key] for r in pts], **kw)


def task2():
    trees = collections.defaultdict(dict)  # run -> problem -> row
    for r in csv.DictReader(open(os.path.join(HERE, "results", "csv", "trees.csv"))):
        if r["log"].startswith("task2/"):
            trees[r["log"][6:]][r["problem_id"]] = r
    chains = collections.defaultdict(list)  # problem -> [(node_id, reward, tokens)]
    nodes = collections.defaultdict(dict)  # (run, problem) -> node_id -> row
    for r in csv.DictReader(open(os.path.join(HERE, "results", "csv", "nodes.csv"))):
        if r["log"] == "task2/b0_64":
            chains[r["problem_id"]].append((int(r["node_id"]), float(r["reward"]), int(r["new_tokens"])))
        elif r["log"].startswith("task2/b"):
            nodes[(r["log"][6:], r["problem_id"])][int(r["node_id"])] = r
    pids = sorted(chains)
    for p in pids:
        chains[p].sort()
    ks = [1, 2, 4, 8, 16, 32, 64]
    c64 = {p: sum(x[1] > 0 for x in chains[p]) for p in pids}
    iid_pass = {k: {p: pass_at_k(64, c64[p], k) for p in pids} for k in ks}
    iid_tok = {k: sum(sum(t for _, _, t in chains[p][:k]) for p in pids) / len(pids) for k in ks}
    mixed = lambda c, k: 1 - math.comb(c, k) / math.comb(64, k) - math.comb(64 - c, k) / math.comb(64, k)  # noqa: E731
    iid_mixed = {k: 100 * sum(mixed(c64[p], k) for p in pids) / len(pids) for k in ks}
    iid_curve = {k: 100 * sum(iid_pass[k].values()) / len(pids) for k in ks}

    def iid_tokens_for(curve_pass, curve_tok, target):  # tokens i.i.d. chains need to reach a PassRate
        for a, b in zip(ks, ks[1:]):
            if curve_pass[a] <= target <= curve_pass[b]:
                f = (target - curve_pass[a]) / (curve_pass[b] - curve_pass[a])
                return math.exp(math.log(curve_tok[a]) + f * (math.log(curve_tok[b]) - math.log(curve_tok[a])))
        return None

    # difficulty from chains 33-64, i.i.d. reference from chains 1-32: independent samples, no selection bias
    c_ref = {p: sum(x[1] > 0 for x in chains[p][:32]) for p in pids}
    c_diff = {p: sum(x[1] > 0 for x in chains[p][32:]) for p in pids}
    ks32 = [1, 2, 4, 8, 16, 32]
    tok32 = {k: iid_tok[k] for k in ks32}
    rng = random.Random(0)
    boot_idx = [[rng.randrange(len(pids)) for _ in pids] for _ in range(1000)]

    rows = []
    for cfg in CONFIGS:
        for method in ("b2", "b1"):
            t = trees.get(f"{method}_{cfg}")
            if not t:
                continue
            tok = sum(int(t[p]["new_tokens"]) for p in pids) / len(pids)
            passes = {p: float(t[p]["pass_any"]) for p in pids}
            row = {"cfg": cfg, "method": method, "leaves": int(t[pids[0]]["n_leaves"]), "tokens": tok,
                   "pass": 100 * sum(passes.values()) / len(pids),
                   "acc": 100 * sum(float(t[p]["acc"]) for p in pids) / len(pids),
                   "distinct": sum(int(t[p]["distinct_answers"]) for p in pids) / len(pids),
                   "mixed": 100 * sum(float(t[p]["mixed"]) for p in pids) / len(pids)}
            iw = interp_w(iid_tok, ks, tok)
            if iw:  # i.i.d. at the same token budget
                lo, hi, w = iw
                m, a, b = boot_ci([passes[p] - ((1 - w) * iid_pass[lo][p] + w * iid_pass[hi][p]) for p in pids])
                row.update(iid=row["pass"] - 100 * m, gain=100 * m, gain_lo=100 * a, gain_hi=100 * b,
                           iid_mixed=(1 - w) * iid_mixed[lo] + w * iid_mixed[hi],
                           iid_distinct=None)
            need = iid_tokens_for(iid_curve, iid_tok, row["pass"])
            if need:  # tokens i.i.d. needs for the same PassRate, with a bootstrap CI over problems
                savings = []
                for idx in boot_idx:
                    bp = {k: 100 * sum(iid_pass[k][pids[i]] for i in idx) / len(idx) for k in ks}
                    bt = {k: sum(sum(x[2] for x in chains[pids[i]][:k]) for i in idx) / len(idx) for k in ks}
                    tr_pass = 100 * sum(passes[pids[i]] for i in idx) / len(idx)
                    tr_tok = sum(int(t[pids[i]]["new_tokens"]) for i in idx) / len(idx)
                    n_ = iid_tokens_for(bp, bt, tr_pass)
                    if n_:
                        savings.append(100 * (1 - tr_tok / n_))
                savings.sort()
                row.update(iid_tokens=need, saving=100 * (1 - tok / need),
                           saving_lo=savings[int(0.025 * len(savings))], saving_hi=savings[int(0.975 * len(savings)) - 1])
            iw32 = interp_w(tok32, ks32, tok)
            if iw32:  # gain by difficulty: tree vs i.i.d. (chains 1-32) at the same tokens, bins from chains 33-64
                lo, hi, w = iw32
                row["by_diff"] = {}
                for name, a_, b_ in DIFF_BINS:
                    ps = [p for p in pids if a_ <= c_diff[p] <= b_]
                    d = [passes[p] - ((1 - w) * pass_at_k(32, c_ref[p], lo) + w * pass_at_k(32, c_ref[p], hi)) for p in ps]
                    row["by_diff"][name] = (len(ps), *[100 * x for x in boot_ci(d)])
            rows.append(row)

    # forks: does the branch flip correctness / change its final answer vs its parent? (all ten tree runs)
    flips = []  # (tree id, surprisal, relative position, flip, answer change, method)
    for (run, p), nd in nodes.items():
        for n in nd.values():
            if n["parent"] == "" or not n["fork_surprisal"]:
                continue
            par = nd[int(n["parent"])]
            a = normalize(n["answer"]) if n["answer"] else None
            b = normalize(par["answer"]) if par["answer"] else None
            flips.append((f"{run}/{p}", float(n["fork_surprisal"]), float(n["fork_rel_pos"]),
                          float(float(n["reward"]) != float(par["reward"])), float(a != b), run[:2]))

    def binned(key_fn, nbins):
        """per-bin flip / answer-change rates with a 95% bootstrap CI over trees (forks of a tree are clustered)"""
        per = collections.defaultdict(lambda: [[0, 0, 0] for _ in range(nbins)])
        for tid, s, pos, f, ch, _ in flips:
            b = key_fn(s, pos)
            per[tid][b][0] += 1
            per[tid][b][1] += f
            per[tid][b][2] += ch
        tids = list(per)
        tot = [[sum(per[t][b][j] for t in tids) for j in range(3)] for b in range(nbins)]
        r = random.Random(1)
        boots = []
        for _ in range(300):
            sample = [tids[r.randrange(len(tids))] for _ in tids]
            sums = [[0, 0, 0] for _ in range(nbins)]
            for t in sample:
                for b in range(nbins):
                    for j in range(3):
                        sums[b][j] += per[t][b][j]
            boots.append(sums)
        out = []
        for b in range(nbins):
            res = {"n": tot[b][0]}
            for j, key in ((1, "flip"), (2, "change")):
                vals = sorted(100 * x[b][j] / x[b][0] for x in boots if x[b][0])
                res[key] = (100 * tot[b][j] / tot[b][0], vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])
            out.append(res)
        return out

    SUR_EDGES = [0, 0.01, 0.1, 0.5, 1.5, 3, 5, 8, 1e9]
    SUR_LABELS = ["<0.01", "0.01–0.1", "0.1–0.5", "0.5–1.5", "1.5–3", "3–5", "5–8", ">8"]
    by_sur = binned(lambda s, pos: next(i for i in range(8) if SUR_EDGES[i] <= s < SUR_EDGES[i + 1]), 8)
    by_pos = binned(lambda s, pos: min(int(pos * 5), 4), 5)
    method_rates = {}
    for m in ("b2", "b1"):
        per = collections.defaultdict(list)
        for tid, s, pos, f, ch, mm in flips:
            if mm == m and any(tid.startswith(f"{m}_{c}/") for c in ("4-1-1-1", "4-3-1-1", "6-2-1-2", "8-4-2-2")):
                per[tid.split("/", 1)[1]].append((f, ch))
        method_rates[m] = {p: (sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)) for p, v in per.items()}
    paired = {}
    for j, key in ((0, "flip"), (1, "change")):
        a, b = method_rates["b2"], method_rates["b1"]
        paired[key] = [100 * x for x in boot_ci([a[p][j] - b[p][j] for p in a if p in b])]
        paired[key + "_levels"] = (100 * sum(v[j] for v in a.values()) / len(a), 100 * sum(v[j] for v in b.values()) / len(b))

    # ---- Figure 1: small multiples (EPTree vs i.i.d., random vs i.i.d.) + gain by difficulty; no overplotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.3, 2.35), gridspec_kw={"width_ratios": [1, 1, 0.95]})
    label_off = {  # direct labels: offset (points) per tree shape, chosen so no label hits a marker or the curve
        "2-1-1-1": (-4, 5, "right"), "2-3-1-1": (-4, 5, "right"), "4-1-1-1": (5, -10, "left"),
        "4-3-1-1": (-4, 5, "right"), "6-2-1-2": (-4, 6, "right"), "8-4-2-2": (-4, 5, "right")}
    for ax, method, color, marker, title in ((ax1, "b2", BLUE, "o", "(a) EPTree vs. i.i.d. chains"),
                                             (ax2, "b1", ORANGE, "s", "(b) random forks vs. i.i.d. chains")):
        ax.axvspan(iid_tok[64] * 1.04, 75000, color=GRID, alpha=0.7, lw=0, zorder=0)
        ax.plot([iid_tok[k] for k in ks], [iid_curve[k] for k in ks], color=AQUA, lw=1.6, ls=":", marker="o", ms=3,
                label="i.i.d. chains", zorder=2)
        pts = sorted((r for r in rows if r["method"] == method), key=lambda r: r["tokens"])
        ax.plot([r["tokens"] for r in pts], [r["pass"] for r in pts], color=color, lw=1.4, ls=":", marker=marker, ms=4.2,
                mec="white", mew=0.6, label="EPTree" if method == "b2" else "random forks", zorder=3)
        for r in pts:
            dx, dy, ha = label_off[r["cfg"]]
            ax.annotate(f"({r['cfg'].replace('-', ',')})", (r["tokens"], r["pass"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=5.6, color=INK2, ha=ha)
        for k in (1, 64):
            ax.annotate(f"$k$={k}", (iid_tok[k], iid_curve[k]), xytext=(3, -9), textcoords="offset points", fontsize=5.6, color=AQUA)
        ax.set_xscale("log")
        ax.set_xlim(600, 75000)
        ax.set_ylim(24, 60)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(kfmt))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("generated tokens per problem")
        ax.set_title(title, loc="left", color=INK)
        ax.legend(frameon=False, loc="upper left", fontsize=6, handlelength=2.2)
        ax.text(60000, 30, "no i.i.d.\nreference", fontsize=5.3, color=INK2, ha="center", va="center")
    ax1.set_ylabel("PassRate (%)")
    ax2.set_yticklabels([])

    names = [n for n, _, _ in DIFF_BINS]
    for method, color, marker, dx in (("b2", BLUE, "o", -0.13), ("b1", ORANGE, "s", 0.13)):
        r = next(r for r in rows if r["cfg"] == "6-2-1-2" and r["method"] == method)
        for i, name in enumerate(names):
            n, m, a_, b_ = r["by_diff"][name]
            ax3.errorbar(i + dx, m, yerr=[[m - a_], [b_ - m]], fmt=marker, color=color, ms=4, capsize=2, lw=1.1,
                         mec="white", mew=0.5, label=("EPTree" if method == "b2" else "random forks") if i == 0 else None)
    ax3.axhline(0, color=INK2, lw=0.8)
    nbin = next(r for r in rows if r["cfg"] == "6-2-1-2")["by_diff"]
    ax3.set_xticks(range(len(DIFF_BINS)))
    ax3.set_xticklabels([f"{n.split()[0]}\n{lo}–{hi}/32\nn={nbin[n][0]}" if lo != hi else f"{n.split()[0]}\n0/32\nn={nbin[n][0]}"
                         for n, lo, hi in DIFF_BINS], fontsize=6)
    ax3.set_ylabel("PassRate gain over i.i.d. (points)")
    ax3.set_title("(c) gain by difficulty, (6,2,1,2)", loc="left", color=INK)
    ax3.legend(frameon=False, loc="upper left", fontsize=6)
    ax3.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=0.6)
    save(fig, "task2_passrate")

    iid_rows = [{"k": k, "tokens": iid_tok[k], "pass": iid_curve[k], "mixed": iid_mixed[k]} for k in ks]
    return rows, iid_rows, by_sur, SUR_LABELS, by_pos, paired


# ------------------------------------------------------------------------------------------------ RL
def load_pp(name):
    d = os.path.join(HERE, "results", "rl", "per_problem")
    for f in (name, name + "_pp"):
        p = os.path.join(d, f + ".jsonl")
        if os.path.exists(p):
            out, seen = {}, collections.Counter()
            for line in open(p):
                r = json.loads(line)
                b = {"MATH500.jsonl": "MATH500", "aimo-validation-amc.jsonl": "AMC"}.get(r["data"], "Omni-MATH-500")
                out[(b, seen[b])] = r["correct"] / r["n"]  # position within its benchmark (AMC ids repeat)
                seen[b] += 1
            return out
    return None


def rl():
    base = load_pp("base_n8")
    run = {"TreeRL": "qwen1.5b-treerl-6-2-1-2-lr5e-6-150", "ChainRL": "qwen1.5b-chainrl-8-lr5e-6-150",
           "GRPO": "qwen1.5b-grpo-8-lr5e-6-150"}
    res = {}
    for m, tag in run.items():
        for s in (50, 100, 150):
            pp = load_pp(f"{tag}__actor_global_step{s}_n8")
            if pp:
                res[(m, s)] = pp

    def diff(a, b, bench=None):
        return [100 * x for x in boot_ci([a[k] - b[k] for k in a if k in b and (bench is None or k[0] == bench)])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.35), gridspec_kw={"width_ratios": [1, 1.1]})
    for m, color, marker in (("TreeRL", BLUE, "o"), ("ChainRL", AQUA, "D"), ("GRPO", ORANGE, "^")):
        steps = [s for s in (50, 100, 150) if (m, s) in res]
        if not steps:
            continue
        d = [diff(res[(m, s)], base) for s in steps]
        off = {"TreeRL": -6, "ChainRL": 6, "GRPO": 0}[m]
        ax1.errorbar([0] + [s + off for s in steps], [0] + [x[0] for x in d],
                     yerr=[[0] + [x[0] - x[1] for x in d], [0] + [x[2] - x[0] for x in d]], color=color, marker=marker,
                     ms=4.5, lw=1.6, capsize=2, mec="white", mew=0.6, label=m + ("" if len(steps) == 3 else " (to step 50 so far)"))
    ax1.axhline(0, color=INK2, lw=0.8)
    ax1.set_xticks([0, 50, 100, 150])
    ax1.set_xlabel("RL training step")
    ax1.set_ylabel("accuracy gain over base (points)")
    ax1.set_title("(a) gain over the base model, 95% CI", loc="left", color=INK)
    ax1.set_ylim(-1.25, 2.6)  # headroom so the legend clears the error bars
    ax1.legend(frameon=False, loc="upper left", ncol=3, fontsize=6, columnspacing=1.0, handlelength=1.6)

    rows = []
    if ("TreeRL", 150) in res and ("ChainRL", 150) in res:
        t, c = res[("TreeRL", 150)], res[("ChainRL", 150)]
        for i, (bench, label) in enumerate([("MATH500", "MATH500 (500)"), ("AMC", "AMC (83)"),
                                            ("Omni-MATH-500", "Omni-MATH-500 (499)"), (None, "all 1,082 pooled")]):
            m, a, b = diff(t, c, bench)
            rows.append((label, m, a, b))
            y = 3 - i
            ax2.errorbar(m, y, xerr=[[m - a], [b - m]], fmt="o", color=BLUE if bench else INK, ms=5, capsize=2.5, lw=1.4,
                         mec="white", mew=0.6)
            ax2.annotate(f"{m:+.2f}  [{a:+.2f}, {b:+.2f}]", (b, y), xytext=(5, 0), textcoords="offset points",
                         fontsize=6.5, color=INK2, va="center")
        ax2.axvline(0, color=INK2, lw=0.8)
        ax2.set_yticks([3, 2, 1, 0])
        ax2.set_yticklabels([r[0] for r in rows])
        ax2.set_xlim(-2, 11)
        ax2.set_xlabel("TreeRL minus ChainRL at step 150 (points)")
        ax2.set_title("(b) head-to-head on the same problems", loc="left", color=INK)
        ax2.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "rl_results")

    table = {}
    for m in run:
        if (m, 150) in res:
            table[m] = {b: 100 * sum(v for k, v in res[(m, 150)].items() if k[0] == b) /
                        sum(1 for k in res[(m, 150)] if k[0] == b) for b in ("MATH500", "AMC", "Omni-MATH-500")}
    table["base"] = {b: 100 * sum(v for k, v in base.items() if k[0] == b) / sum(1 for k in base if k[0] == b)
                     for b in ("MATH500", "AMC", "Omni-MATH-500")}
    vs_base = {(m, s): diff(res[(m, s)], base) for (m, s) in res}
    return rows, table, vs_base


def main():
    rows, iid_rows, by_sur, sur_labels, by_pos, paired = task2()
    print("== Task 2 (500 Omni-MATH problems)")
    for r in iid_rows:
        print(f"  iid k={r['k']:2d} tokens {r['tokens']:7.0f} pass {r['pass']:.1f} mixed {r['mixed']:.1f}")
    for r in rows:
        line = (f"  {r['method']} {r['cfg']:8s} leaves {r['leaves']:3d} tokens {r['tokens']:7.0f} pass {r['pass']:.1f} "
                f"acc {r['acc']:.1f} distinct {r['distinct']:.2f} mixed {r['mixed']:.1f}")
        if "gain" in r:
            line += f" | iid@tok {r['iid']:.1f} gain {r['gain']:+.1f} [{r['gain_lo']:+.1f},{r['gain_hi']:+.1f}] iid mixed {r['iid_mixed']:.1f}"
        if "saving" in r:
            line += f" | iid tokens {r['iid_tokens']:.0f} saving {r['saving']:.0f}% [{r['saving_lo']:.0f},{r['saving_hi']:.0f}]"
        print(line)
        if "by_diff" in r:
            print("      by difficulty: " + "  ".join(f"{k} n={v[0]} {v[1]:+.1f} [{v[2]:+.1f},{v[3]:+.1f}]" for k, v in r["by_diff"].items()))
    for lab, d in zip(sur_labels, by_sur):
        print(f"  surprisal {lab:9s} n={d['n']:6d} flip {d['flip'][0]:.1f} [{d['flip'][1]:.1f},{d['flip'][2]:.1f}]  change {d['change'][0]:.1f}")
    for i, d in enumerate(by_pos):
        print(f"  position q{i} n={d['n']:6d} flip {d['flip'][0]:.1f} [{d['flip'][1]:.1f},{d['flip'][2]:.1f}]  change {d['change'][0]:.1f}")
    for key in ("change", "flip"):
        print(f"  {key}: EPTree {paired[key + '_levels'][0]:.1f} random {paired[key + '_levels'][1]:.1f}  paired {paired[key][0]:+.1f} [{paired[key][1]:+.1f},{paired[key][2]:+.1f}]")
    hh, table, vs_base = rl()
    print("== RL (8 samples per problem)")
    for m, t in table.items():
        print(f"  {m:8s} " + "  ".join(f"{b} {v:.2f}" for b, v in t.items()) + f"  mean-of-3 {sum(t.values()) / 3:.2f}")
    for k, v in sorted(vs_base.items()):
        print(f"  {k[0]:8s} @{k[1]:3d} vs base {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]")
    for label, m, a, b in hh:
        print(f"  TreeRL-ChainRL @150 {label:22s} {m:+.2f} [{a:+.2f}, {b:+.2f}]")
    print("figures in", OUT)


if __name__ == "__main__":
    main()
