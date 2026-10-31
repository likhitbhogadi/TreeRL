import json, sys, os
sys.path.insert(0, os.getcwd())
from tree_alloc.metrics import paired_bootstrap
def load(p):
    d = {}; cnt = {}
    for l in open(p):
        r = json.loads(l); k = cnt.get(r["data"], 0); cnt[r["data"]] = k + 1
        d[f"{r['data']}/{k}:{r['id']}"] = r["correct"] / r["n"]
    return d
RUNS = {"TreeRL": "treerl-6-2-1-2", "ChainRL": "chainrl-16", "Ours main (4,2,1,2) k10 h8": "ours-4-2-1-2-k10",
        "Abl1 (2,2,1,2) k20 h8": "ours-2-2-1-2-k20", "Abl2 (4,1,1,3) k14 h20": "ours-4-1-1-3-k14",
        "Abl3 (4,2,1,2) k10 h8 lam0.3": "ours-4-2-1-2-k10-h8-lam0.3"}
SETS = ["MATH500", "aimo-validation-amc", "omni_math_500_seed0", "olympiad_bench"]
for mode, sub in (("greedy", "rl_paper_per_problem_greedy"), ("n8", "rl_paper_per_problem")):
    p = lambda m, s: f"results/{sub}/qwen1.5b-paper-{m}-seed42__actor_global_step{s}_{mode}.jsonl"
    b = load(f"results/{sub}/base_{mode}.jsonl")
    def per(a): return " ".join(f"{100*sum(v for k,v in a.items() if k.startswith(s+'.jsonl/') or k.startswith(s+'/'))/max(1,sum(1 for k in a if k.startswith(s+'.jsonl/') or k.startswith(s+'/'))):5.1f}" for s in SETS)
    print(f"\n== {mode}   base {100*sum(b.values())/len(b):.2f}  [{per(b)}]")
    for s in (50, 100, 125, 150):
        # no TreeRL/ChainRL checkpoint at step 125 (ablation 3 stopped there): compare it with their step 150
        rs = s if os.path.exists(p("treerl-6-2-1-2", s)) else 150
        T = load(p("treerl-6-2-1-2", rs)); C = load(p("chainrl-16", rs))
        for name, m in RUNS.items():
            if not os.path.exists(p(m, s)): continue
            a = load(p(m, s)); row = f"  step {s:3d} {name:28s} {100*sum(a.values())/len(a):6.2f} [{per(a)}]"
            for rn, ref in (("base", b), ("TreeRL" + ("" if rs == s else f"@{rs}"), T), ("ChainRL" + ("" if rs == s else f"@{rs}"), C)):
                if ref is a or (rn.startswith("TreeRL") and m.startswith("treerl")) or (rn.startswith("ChainRL") and m.startswith("chainrl")): continue
                bs = paired_bootstrap(a, ref); sig = "*" if bs["lo"] > 0 or bs["hi"] < 0 else " "
                row += f" | {rn} {100*bs['diff']:+.2f} [{100*bs['lo']:+.2f},{100*bs['hi']:+.2f}]{sig}"
            print(row)
