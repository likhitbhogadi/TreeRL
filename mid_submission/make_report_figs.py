"""Figures for to_report.md. Run from mid_submission/:  python3 make_report_figs.py
Reads the per-problem greedy files and writes, under report/figures/:
  fig_greedy_vs_step.{pdf,png}   pooled greedy accuracy over training, all runs
  fig_main_vs_treerl.{pdf,png}   per-benchmark gain of our main run (step 150) over TreeRL
  fig_omni_math.{pdf,png}        Omni-MATH greedy accuracy per checkpoint"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = "results/rl_paper_per_problem_greedy"
OUT = "report/figures"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
# label, file key, steps, color, marker, linestyle, linewidth
RUNS = [
    ("ChainRL", "chainrl-16", (50, 100, 150, 200), "#a3a29c", "s", "--", 1.8),
    ("TreeRL", "treerl-6-2-1-2", (50, 100, 150, 200), "#52514e", "D", "--", 1.8),
    ("Ours (main)", "ours-4-2-1-2-k10", (50, 100, 150), "#2a78d6", "o", "-", 2.6),
    ("Ours (abl. 1)", "ours-2-2-1-2-k20", (50, 100, 150), "#eb6834", "^", "-", 1.5),
    ("Ours (abl. 2)", "ours-4-1-1-3-k14", (50, 100, 150), "#1baf7a", "v", "-", 1.5),
    ("Ours (abl. 3)", "ours-4-2-1-2-k10-h8-lam0.3", (50, 100, 125), "#eda100", "P", "-", 1.5),
]
BENCH = [("MATH500", "MATH500"), ("AMC", "aimo"), ("Omni-MATH", "omni"), ("OlympiadBench", "olympiad")]


def acc(path, prefix=""):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["data"].startswith(prefix)]
    return 100 * sum(r["correct"] / r["n"] for r in rows) / len(rows)


def ckpt(key, step):
    return f"{G}/qwen1.5b-paper-{key}-seed42__actor_global_step{step}_greedy.jsonl"


plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})
os.makedirs(OUT, exist_ok=True)
BASE = f"{G}/base_greedy.jsonl"


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=200)
    plt.close(fig)


# Figure 1: pooled greedy accuracy vs training step
base = acc(BASE)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for label, key, steps, color, marker, ls, lw in RUNS:
    xs, ys = [0, *steps], [base, *(acc(ckpt(key, s)) for s in steps)]
    ax.plot(xs, ys, ls, color=color, marker=marker, markersize=6, linewidth=lw, label=label,
            markeredgecolor="white", markeredgewidth=1, zorder=4 if "main" in label else 2)
best = acc(ckpt("ours-4-2-1-2-k10", 150))
ax.annotate(f"{best:.2f}", (150, best), xytext=(7, 4), textcoords="offset points",
            fontsize=9, color=INK, fontweight="bold")
ax.set_xticks([0, 50, 100, 125, 150, 200])
ax.set_xlabel("Training step")
ax.set_ylabel("Greedy accuracy, 1,755 problems (%)")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
ax.set_ylim(46.6, 48.95)
save(fig, "fig_greedy_vs_step")

# Figure 2: gain of our main run @150 over TreeRL, per benchmark (diverging bars from 0)
main = ckpt("ours-4-2-1-2-k10", 150)
comps = [("vs TreeRL, step 150", ckpt("treerl-6-2-1-2", 150), "#2a78d6"),
         ("vs TreeRL, final step 200", ckpt("treerl-6-2-1-2", 200), "#86b6ef")]
rows = [*BENCH, ("All 1,755", "")]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
h = 0.36
for j, (clabel, ref, color) in enumerate(comps):
    for i, (bname, pre) in enumerate(rows):
        d = acc(main, pre) - acc(ref, pre)
        y = i + (j - 0.5) * (h + 0.04)
        ax.barh(y, d, height=h, color=color, label=clabel if i == 0 else None)
        ax.annotate(f"{d:+.1f}" if pre else f"{d:+.2f}", (max(d, 0), y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)
ax.axvline(0, color=MUTED, linewidth=1)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.invert_yaxis()
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(-0.3, 3.0)
ax.set_xlabel("Greedy accuracy gain of our main run (step 150), points")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
save(fig, "fig_main_vs_treerl")

# Figure 3: Omni-MATH greedy accuracy per checkpoint (dot plot)
groups = [("Step 100", {r[0]: 100 for r in RUNS}),
          ("Step 150 (abl. 3: step 125)", {**{r[0]: 150 for r in RUNS}, "Ours (abl. 3)": 125}),
          ("Final (step 200)", {"ChainRL": 200, "TreeRL": 200})]
fig, ax = plt.subplots(figsize=(6.6, 5.6))
y, yt, yl = 0, [], []
for gi, (gname, stepmap) in enumerate(groups):
    if gi:
        ax.axhline(y + 0.4, color=GRID, linewidth=0.8)
    ax.text(25.85, y, gname, fontsize=9, fontweight="bold", color=INK, va="center")
    y -= 1
    for label, key, steps, color, marker, _, _ in RUNS:
        step = stepmap.get(label)
        if step not in steps:
            continue
        v = acc(ckpt(key, step), "omni")
        ax.plot([v], [y], marker=marker, color=color, markersize=9, markeredgecolor="white",
                markeredgewidth=1, linestyle="none")
        ax.annotate(f"{v:.1f}", (v, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
        yt.append(y); yl.append(label)
        y -= 1
    y -= 0.2
omni_base = acc(BASE, "omni")
ax.axvline(omni_base, color=MUTED, linestyle=":", linewidth=1.5)
ax.text(omni_base, 0.75, f"starting model {omni_base:.1f}", ha="center", fontsize=8.5, color=INK2)
ax.set_yticks(yt, yl)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(25.8, 31)
ax.set_ylim(y + 0.5, 1.2)
ax.set_xlabel("Omni-MATH greedy accuracy (%)")
save(fig, "fig_omni_math")
print("wrote", OUT)
