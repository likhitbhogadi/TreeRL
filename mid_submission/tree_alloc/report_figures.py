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
def task2():
    trees = collections.defaultdict(dict)  # log -> problem -> row
    for r in csv.DictReader(open(os.path.join(HERE, "results", "csv", "trees.csv"))):
        if r["log"].startswith("task2/"):
            trees[r["log"][6:]][r["problem_id"]] = r
    chains = collections.defaultdict(list)  # problem -> [(node_id, reward, tokens)]
    forks = collections.defaultdict(lambda: collections.defaultdict(dict))  # log -> problem -> node_id -> row
    for r in csv.DictReader(open(os.path.join(HERE, "results", "csv", "nodes.csv"))):
        if r["log"] == "task2/b0_64":
            chains[r["problem_id"]].append((int(r["node_id"]), float(r["reward"]), int(r["new_tokens"])))
        elif r["log"].startswith("task2/b"):
            forks[r["log"][6:]][r["problem_id"]][int(r["node_id"])] = r
    pids = sorted(chains)
    ks = [1, 2, 4, 8, 16, 32, 64]
    iid_pass = {k: {p: pass_at_k(64, sum(c[1] > 0 for c in chains[p]), k) for p in pids} for k in ks}
    iid_tok = {k: sum(sum(t for i, _, t in sorted(chains[p])[:k]) for p in pids) / len(pids) for k in ks}

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
                   "distinct": sum(int(t[p]["distinct_answers"]) for p in pids) / len(pids)}
            lo = max((k for k in ks if iid_tok[k] <= tok), default=None)
            hi = min((k for k in ks if iid_tok[k] >= tok), default=None)
            if lo and hi:  # i.i.d. PassRate at the same token budget, log-linear between neighbouring k
                w = 0 if lo == hi else (math.log(tok) - math.log(iid_tok[lo])) / (math.log(iid_tok[hi]) - math.log(iid_tok[lo]))
                diffs = [passes[p] - ((1 - w) * iid_pass[lo][p] + w * iid_pass[hi][p]) for p in pids]
                m, a, b = boot_ci(diffs)
                row.update(iid=row["pass"] - 100 * m, gain=100 * m, gain_lo=100 * a, gain_hi=100 * b)
            rows.append(row)

    # fork outcomes: does a branch end in a different final answer / different correctness than its parent?
    fork = {}
    for method in ("b2", "b1"):
        change, flip, sur = collections.defaultdict(list), collections.defaultdict(list), []
        for cfg in ("4-1-1-1", "4-3-1-1", "6-2-1-2", "8-4-2-2"):  # the four shapes run with both methods
            for p, nodes in forks[f"{method}_{cfg}"].items():
                for n in nodes.values():
                    if n["parent"] == "":
                        continue
                    par = nodes[int(n["parent"])]
                    a = normalize(n["answer"]) if n["answer"] else None
                    b = normalize(par["answer"]) if par["answer"] else None
                    change[p].append(float(a != b))
                    flip[p].append(float(float(n["reward"]) != float(par["reward"])))
                    if n["fork_surprisal"]:
                        sur.append(float(n["fork_surprisal"]))
        fork[method] = {"change": {p: sum(v) / len(v) for p, v in change.items()},
                        "flip": {p: sum(v) / len(v) for p, v in flip.items()}, "surprisal": sur}

    # ---- Figure: PassRate vs tokens, and gain over i.i.d. at matched tokens
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.35), gridspec_kw={"width_ratios": [1.05, 1]})
    ax1.plot([iid_tok[k] for k in ks], [100 * sum(iid_pass[k].values()) / len(pids) for k in ks], color=AQUA, lw=2,
             marker="o", ms=4, label="i.i.d. chains (pass@$k$)", zorder=2)
    for k in (4, 16, 64):
        ax1.annotate(f"$k$={k}", (iid_tok[k], 100 * sum(iid_pass[k].values()) / len(pids)), xytext=(4, -9),
                     textcoords="offset points", fontsize=6.5, color=INK2)
    for method, color, marker, name in (("b2", BLUE, "o", "EPTree (entropy forks)"), ("b1", ORANGE, "s", "random forks")):
        pts = [r for r in rows if r["method"] == method]
        ax1.scatter([r["tokens"] for r in pts], [r["pass"] for r in pts], color=color, marker=marker, s=26,
                    edgecolors="white", linewidths=0.8, label=name, zorder=3)
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(kfmt))
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xlabel("generated tokens per problem (log scale)")
    ax1.set_ylabel("PassRate (%)")
    ax1.set_title("(a) PassRate vs. generation budget", loc="left", color=INK)
    ax1.legend(frameon=False, loc="lower right")

    matched = [r for r in rows if "gain" in r]
    order = [c for c in CONFIGS if any(r["cfg"] == c for r in matched)]
    for method, color, marker, dx in (("b2", BLUE, "o", -0.13), ("b1", ORANGE, "s", 0.13)):
        for i, cfg in enumerate(order):
            r = next((r for r in matched if r["cfg"] == cfg and r["method"] == method), None)
            if r:
                ax2.errorbar(i + dx, r["gain"], yerr=[[r["gain"] - r["gain_lo"]], [r["gain_hi"] - r["gain"]]],
                             fmt=marker, color=color, ms=4.5, capsize=2, lw=1.2, mec="white", mew=0.6)
    ax2.axhline(0, color=INK2, lw=0.8)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels([f"({c.replace('-', ',')})\n{kfmt(next(r['tokens'] for r in matched if r['cfg'] == c))} tok"
                         for c in order], fontsize=6.5)
    ax2.set_ylabel("PassRate gain over i.i.d. (points)")
    ax2.set_title("(b) gain at the same token budget, 95% CI", loc="left", color=INK)
    ax2.plot([], [], "o", color=BLUE, label="EPTree")
    ax2.plot([], [], "s", color=ORANGE, label="random forks")
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "task2_passrate")

    # ---- Figure: where EPTree forks, and what a fork changes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.1, 3.6), gridspec_kw={"height_ratios": [1, 1]})
    for method, color, name in (("b2", BLUE, "EPTree"), ("b1", ORANGE, "random")):
        s = sorted(max(x, 1e-4) for x in fork[method]["surprisal"])
        ax1.plot(s, [(i + 1) / len(s) for i in range(len(s))], color=color, lw=2, label=name)
    eptree_med = sorted(fork["b2"]["surprisal"])[len(fork["b2"]["surprisal"]) // 2]
    near0 = sum(x < 1e-3 for x in fork["b1"]["surprisal"]) / len(fork["b1"]["surprisal"])
    ax1.annotate(f"EPTree\nmedian {eptree_med:.2f} nats", (eptree_med, 0.5), xytext=(-8, 0), textcoords="offset points",
                 fontsize=6.5, color=INK2, ha="right", va="center")
    ax1.text(1.6e-4, 0.32, f"random: {100 * near0:.0f}% of forks at\ntokens with $-\\log p$ < 0.001", fontsize=6.5,
             color=INK2, ha="left", va="center")
    ax1.set_xscale("log")
    ax1.set_xlim(1e-4, 30)
    ax1.set_xlabel(r"surprisal of the token forked at, $-\log p$ (nats, log scale)")
    ax1.set_ylabel("share of forks")
    ax1.set_title("(a) EPTree forks at far more surprising tokens", loc="left", color=INK)
    labels = ["final answer\nchanges", "correctness\nchanges"]
    for j, key in enumerate(("change", "flip")):
        for method, color, marker, dx in (("b2", BLUE, "o", -0.12), ("b1", ORANGE, "s", 0.12)):
            m, a, b = boot_ci(list(fork[method][key].values()))
            ax2.errorbar(j + dx, 100 * m, yerr=[[100 * (m - a)], [100 * (b - m)]], fmt=marker, color=color, ms=4.5,
                         capsize=2, lw=1.2, mec="white", mew=0.6)
            ax2.annotate(f"{100 * m:.1f}%", (j + dx, 100 * m), xytext=(5 if dx > 0 else -5, 0), textcoords="offset points",
                         fontsize=6.5, color=INK2, va="center", ha="left" if dx > 0 else "right")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(labels)
    ax2.set_xlim(-0.6, 1.6)
    ax2.set_ylim(0, 60)
    ax2.set_ylabel("share of branches (%)")
    ax2.set_title("(b) branch vs. its parent, 95% CI", loc="left", color=INK)
    ax2.plot([], [], "o", color=BLUE, label="EPTree")
    ax2.plot([], [], "s", color=ORANGE, label="random")
    ax2.legend(frameon=False, loc="upper right")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout(h_pad=1.2)
    save(fig, "task2_forks")

    # paired EPTree - random difference in fork outcomes (same problems)
    paired = {}
    for key in ("change", "flip"):
        a, b = fork["b2"][key], fork["b1"][key]
        paired[key] = [100 * x for x in boot_ci([a[p] - b[p] for p in a if p in b])]
    return rows, fork, paired


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
        off = {"TreeRL": -3, "ChainRL": 3, "GRPO": 0}[m]
        ax1.errorbar([0] + [s + off for s in steps], [0] + [x[0] for x in d],
                     yerr=[[0] + [x[0] - x[1] for x in d], [0] + [x[2] - x[0] for x in d]], color=color, marker=marker,
                     ms=4.5, lw=1.6, capsize=2, mec="white", mew=0.6, label=m + ("" if len(steps) == 3 else " (to step 50 so far)"))
    ax1.axhline(0, color=INK2, lw=0.8)
    ax1.set_xticks([0, 50, 100, 150])
    ax1.set_xlabel("RL training step")
    ax1.set_ylabel("accuracy gain over base (points)")
    ax1.set_title("(a) gain over the base model, 95% CI", loc="left", color=INK)
    ax1.legend(frameon=False, loc="upper left")

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
    rows, fork, paired = task2()
    print("== Task 2 (500 Omni-MATH problems)")
    for r in rows:
        g = f"  gain {r['gain']:+.1f} [{r['gain_lo']:+.1f}, {r['gain_hi']:+.1f}]  iid {r['iid']:.1f}" if "gain" in r else ""
        print(f"  {r['method']} {r['cfg']:8s} leaves {r['leaves']:3d} tokens {r['tokens']:7.0f} pass {r['pass']:.1f} distinct {r['distinct']:.2f}{g}")
    for m in ("b2", "b1"):
        s = sorted(fork[m]["surprisal"])
        print(f"  forks {m}: n={len(s)} median surprisal {s[len(s) // 2]:.2f} mean {sum(s) / len(s):.2f}")
    for key in ("change", "flip"):
        for m in ("b2", "b1"):
            v = boot_ci(list(fork[m][key].values()))
            print(f"  {key:6s} {m}: {100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]")
        print(f"  {key:6s} EPTree - random (paired): {paired[key][0]:+.1f} [{paired[key][1]:+.1f}, {paired[key][2]:+.1f}]")
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
