"""Paper-sized RL figures for anlp_Final_submission_draft/acl_latex.tex.
Run from mid_submission/:  python3 make_paper_figs.py
Writes, under anlp_Final_submission_draft/figures/:
  rl_main.pdf    full width: (a) pooled greedy accuracy vs step, (b) gain of our main run over both baselines at step 150
  rl_omni.pdf    one column: Omni-MATH greedy accuracy per checkpoint"""
import json, os, sys
sys.path.insert(0, os.getcwd())
from tree_alloc.metrics import paired_bootstrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = "results/rl_paper_per_problem_greedy"
OUT = "anlp_Final_submission_draft/figures"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
RUNS = [  # label, file key, steps, color, marker, linestyle, linewidth
    ("ChainRL", "chainrl-16", (50, 100, 150), "#a3a29c", "s", "--", 1.4),
    ("TreeRL", "treerl-6-2-1-2", (50, 100, 150), "#52514e", "D", "--", 1.4),
    ("Ours (main)", "ours-4-2-1-2-k10", (50, 100, 150), "#2a78d6", "o", "-", 2.2),
    ("Ours (abl. 1)", "ours-2-2-1-2-k20", (50, 100, 150), "#eb6834", "^", "-", 1.2),
    ("Ours (abl. 2)", "ours-4-1-1-3-k14", (50, 100, 150), "#1baf7a", "v", "-", 1.2),
    ("Ours (abl. 3)", "ours-4-2-1-2-k10-h8-lam0.3", (50, 100), "#eda100", "P", "-", 1.2),
]
BENCH = [("MATH500", "MATH500"), ("AMC", "aimo"), ("Omni-MATH", "omni"), ("OlympiadBench", "olympiad"), ("All", "")]


def acc(path, prefix=""):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["data"].startswith(prefix)]
    return 100 * sum(r["correct"] / r["n"] for r in rows) / len(rows)


def per_problem(path, prefix=""):
    """Per-problem scores keyed the same way as final_ci.py, for paired bootstrap CIs."""
    d, cnt = {}, {}
    for l in open(path):
        r = json.loads(l)
        if not r["data"].startswith(prefix):
            continue
        k = cnt.get(r["data"], 0)
        cnt[r["data"]] = k + 1
        d[f"{r['data']}/{k}:{r['id']}"] = r["correct"] / r["n"]
    return d


def ckpt(key, step):
    return f"{G}/qwen1.5b-paper-{key}-seed42__actor_global_step{step}_greedy.jsonl"


plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
os.makedirs(OUT, exist_ok=True)
BASE = f"{G}/base_greedy.jsonl"

# rl_main: (a) greedy vs step, (b) gain over TreeRL per benchmark
fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.3, 2.35), gridspec_kw={"width_ratios": [1.25, 1]})
base = acc(BASE)
ax.grid(axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for label, key, steps, color, marker, ls, lw in RUNS:
    xs, ys = [0, *steps], [base, *(acc(ckpt(key, s)) for s in steps)]
    # Step-150 pooled accuracy confirmed by the later ablation-3 run.
    if key == "ours-4-2-1-2-k10-h8-lam0.3":
        xs.append(150)
        ys.append(47.98)
    ax.plot(xs, ys, ls, color=color, marker=marker, markersize=4, linewidth=lw, label=label,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4 if "main" in label else 2)
best = acc(ckpt("ours-4-2-1-2-k10", 150))
ax.annotate(f"{best:.2f}", (150, best), xytext=(-30, 3), textcoords="offset points",
            fontsize=7, color=INK, fontweight="bold")
ax.set_xticks([0, 50, 100, 150])
ax.set_xlabel("Training step")
ax.set_ylabel("Greedy accuracy (%)")
ax.set_ylim(46.6, 49.35)
ax.legend(frameon=False, fontsize=6.3, loc="upper left", ncol=3, handlelength=2.2, columnspacing=1.0)
ax.set_title("(a) All 1,755 problems", fontsize=8, loc="left", color=INK)

main = ckpt("ours-4-2-1-2-k10", 150)
comps = [("vs TreeRL", ckpt("treerl-6-2-1-2", 150), "#2a78d6"),
         ("vs ChainRL", ckpt("chainrl-16", 150), "#86b6ef")]
h = 0.36
XLO, XHI = -5.5, 12.5
for j, (clabel, ref, color) in enumerate(comps):
    for i, (bname, pre) in enumerate(BENCH):
        # paired bootstrap over problems (same resampling as tree_alloc.metrics.paired_bootstrap)
        bs = paired_bootstrap(per_problem(main, pre), per_problem(ref, pre))
        d, lo, hi = 100 * bs["diff"], 100 * bs["lo"], 100 * bs["hi"]
        y = i + (j - 0.5) * (h + 0.04)
        bx.barh(y, d, height=h, color=color, label=clabel if i == 0 else None)
        bx.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="none", ecolor=INK2, elinewidth=0.7,
                    capsize=1.6, capthick=0.7)
        bx.annotate(f"{d:+.1f}" if pre else f"{d:+.2f}", (hi, y), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.3, color=INK2)
bx.axvline(0, color=MUTED, linewidth=0.8)
bx.set_yticks(range(len(BENCH)), [b[0] for b in BENCH])
bx.invert_yaxis()
bx.tick_params(axis="y", length=0)
bx.spines["left"].set_visible(False)
bx.set_xlim(XLO, XHI)
bx.set_xlabel("Gain of ours (main, step 150), points")
bx.legend(frameon=False, fontsize=6.3, loc="center right", bbox_to_anchor=(1.0, 0.36), handlelength=1.5)
bx.set_title("(b) Gain of ours at step 150", fontsize=8, loc="left", color=INK)
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{OUT}/rl_main.pdf")
plt.close(fig)

# rl_omni: Omni-MATH dot plot, one column
groups = [("Step 100", {r[0]: 100 for r in RUNS}), ("Step 150", {r[0]: 150 for r in RUNS})]
fig, ax = plt.subplots(figsize=(3.05, 2.75))
y, yt, yl = 0, [], []
for gi, (gname, stepmap) in enumerate(groups):
    if gi:
        ax.axhline(y + 0.4, color=GRID, linewidth=0.6)
    ax.text(25.85, y, gname, fontsize=7, fontweight="bold", color=INK, va="center")
    y -= 1
    for label, key, steps, color, marker, _, _ in RUNS:
        step = stepmap.get(label)
        if step not in steps:
            continue
        v = acc(ckpt(key, step), "omni")
        ax.plot([v], [y], marker=marker, color=color, markersize=6, markeredgecolor="white",
                markeredgewidth=0.6, linestyle="none")
        ax.annotate(f"{v:.1f}", (v, y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color=INK2)
        yt.append(y); yl.append(label)
        y -= 1
    y -= 0.2
omni_base = acc(BASE, "omni")
ax.axvline(omni_base, color=MUTED, linestyle=":", linewidth=1)
ax.text(omni_base - 0.08, 0.8, f"starting model ({omni_base:.1f})", ha="right", fontsize=6.5, color=INK2)
ax.set_yticks(yt, yl)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(25.8, 30.6)
ax.set_ylim(y + 0.5, 1.3)
ax.set_xlabel("Omni-MATH greedy accuracy (%)")
fig.tight_layout()
fig.savefig(f"{OUT}/rl_omni.pdf")
plt.close(fig)
print("wrote", OUT)
