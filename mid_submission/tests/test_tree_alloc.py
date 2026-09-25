"""Run from mid_submission/:  python3 -m unittest discover -s tests -v"""
import json
import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tree_alloc import methods as M  # noqa: E402
from tree_alloc.data import load_problems  # noqa: E402
from tree_alloc.gen import FakeGenerator  # noqa: E402
from tree_alloc.metrics import paired_bootstrap, spearman, tree_metrics  # noqa: E402
from tree_alloc.oracle import candidate_states, run_oracle  # noqa: E402
from tree_alloc.run import main  # noqa: E402
from tree_alloc.segments import SegTree, treerl_step_rewards  # noqa: E402
from tree_alloc.tree import Generation, Tree, annotate  # noqa: E402
from tree_alloc.verify import Verifier, extract_boxed  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FG = FakeGenerator()
ID = FG.id_of


def toks(*words):
    return [ID[w] for w in words]


def gen(ids, lp=-0.5, finish="stop"):
    return Generation(list(ids), [lp] * len(ids), finish)


def fake_trees(n_problems=3, seed=0):
    g = FakeGenerator(seed=seed)
    return g, [Tree(f"q{i}", g.encode_prompt(f"question {i}"), "7") for i in range(n_problems)]


def check_consistency(tc, t):
    """Every (node, pos) fork point reproduces the trajectory prefix."""
    for n in t.nodes:
        ids, _ = t.response(n.id)
        for pos in range(0, len(ids) + 1, 3):
            parent, idx = t.locate(n.id, pos)
            tc.assertEqual(t.prefix_ids(parent, idx), ids[:pos])


class TestAnnotate(unittest.TestCase):
    def test_boundaries_and_boxed(self):
        text = {0: "a", 1: ".\n\n", 2: "b", 3: "\n", 4: "\n", 5: "c", 6: "\\", 7: "boxed", 8: "{"}
        seg_ends, boxed = annotate(list(range(9)), text.get)
        self.assertEqual(seg_ends, [2, 5])  # after ".\n\n" and after the split "\n","\n"
        self.assertEqual(boxed, 6)  # start of "\boxed", not the token completing it

    def test_newline_run_is_one_boundary(self):
        text = {0: "a", 1: "\n\n", 2: "\n", 3: "b"}
        self.assertEqual(annotate([0, 1, 2, 3], text.get)[0], [3])


class TestTree(unittest.TestCase):
    def setUp(self):
        self.t = Tree("p", [101, 102], "7")
        s1 = toks("Let", " x", ".", "\n\n")
        s2 = toks(" so", " we", ".", "\n\n")
        ans = toks("\\", "boxed", "{", "7", "}")
        self.t.add(None, 0, gen(s1 + s2 + ans), FG.token_text)  # node 0
        # node 1 branches at the second segment boundary-free position 4 (start of step 2)
        self.t.add(0, 4, gen(toks(" get", ".", "\n\n") + toks("\\", "boxed", "{", "3", "}")), FG.token_text)
        # node 2 branches inside node 1
        self.t.add(1, 3, gen(toks("\\", "boxed", "{", "7", "}")), FG.token_text)

    def test_prefix_and_locate(self):
        check_consistency(self, self.t)
        self.assertEqual(self.t.nodes[2].offset, 7)
        self.assertEqual(self.t.boundaries(2), [4, 7, 12])
        self.assertEqual(self.t.boxed_pos(0), 8)
        self.assertEqual(self.t.new_tokens, 13 + 8 + 5)

    def test_json_roundtrip(self):
        t2 = Tree.from_json(json.loads(json.dumps(self.t.to_json())))
        self.assertEqual(t2.response(2), self.t.response(2))

    def test_segments_values_ce(self):
        for n, r in zip(self.t.nodes, [1.0, 0.0, 1.0]):
            n.reward = r
        seg = SegTree(self.t)
        # every leaf's segment path reconstructs its response
        for leaf in range(3):
            v = next(v for v in seg.nodes if leaf in v.ends)
            path = []
            while v is not None:
                path.append(v)
                v = v.parent
            self.assertEqual(path[0].end_pos, len(self.t.response(leaf)[0]))
        root, first = seg.root, seg.root.children[0]
        self.assertEqual(len(root.children), 1)
        self.assertEqual(seg.delta(first), 0.0)  # single child => delta == 0 exactly
        bp = seg.branch_points()
        self.assertEqual([len(u.children) for u in bp], [2, 2])
        self.assertAlmostEqual(seg.value(root), 2 / 3)
        mu, sd, k = seg.ce_stats(first)
        self.assertAlmostEqual(mu, 0.5)  # constant logprob -0.5
        self.assertAlmostEqual(sd, 0.0)
        self.assertEqual(k, 3)
        rew = treerl_step_rewards(seg)
        for v in seg.nodes[1:]:
            g = seg.value(v) - seg.value(root)
            l = seg.value(v) - seg.value(v.parent)
            self.assertAlmostEqual(rew[v.idx], (g + l) / math.sqrt(len(v.leaves)))


class TestMethods(unittest.TestCase):
    def test_b0(self):
        g, trees = fake_trees()
        M.run_iid(trees, g, 4, 3000, 4096)
        self.assertTrue(all(len(t.nodes) == 4 and all(n.parent is None for n in t.nodes) for t in trees))

    def test_eptree_counts_and_eligibility(self):
        for random_fork in (False, True):
            g, trees = fake_trees()
            rngs = [random.Random(i) for i in range(len(trees))]
            M.run_eptree(trees, g, rngs, M=3, N=2, L=2, T=2, max_response=3000, max_model_len=4096,
                         random_fork=random_fork)
            for t in trees:
                self.assertEqual(len(t.nodes), 3 + 2 * 3 * 2 * 2)  # M + L*M*N*T (top-N per chain)
                # every chain got N*T new leaves per iteration
                roots = [M.root_of(t, n.id) for n in t.nodes if n.parent is not None]
                self.assertEqual(sorted(roots.count(r) for r in range(3)), [8, 8, 8])
                for n in t.nodes:
                    if n.parent is None:
                        continue
                    p = t.nodes[n.parent]
                    self.assertGreaterEqual(n.fork_idx, 1)
                    self.assertLess(n.offset, 0.9 * p.end)
                    bp = t.boxed_pos(p.id)
                    self.assertTrue(bp is None or n.offset <= bp)
                check_consistency(self, t)

    def test_eptree_picks_highest_surprisal(self):
        g, trees = fake_trees(1)
        rngs = [random.Random(0)]
        M.run_eptree(trees, g, rngs, M=2, N=1, L=1, T=1, max_response=3000, max_model_len=4096)
        t = trees[0]
        roots_only = Tree.from_json(t.to_json())
        roots_only.nodes = roots_only.nodes[:2]
        cands = M.eptree_candidates(roots_only, set(), 0.1)
        for child in t.nodes[2:]:  # one fork per chain, at that chain's max-surprisal token
            best = max(c[0] for c in cands[child.parent])
            self.assertEqual(-t.nodes[child.parent].logprobs[child.fork_idx], best)

    def test_p1_budget_and_segment_forks(self):
        for score in M.P1_SCORES:
            g, trees = fake_trees()
            rngs = [random.Random(i) for i in range(len(trees))]
            M.run_p1(trees, g, rngs, budget=6, max_response=3000, max_model_len=4096, score=score,
                     min_suffix=4)
            for t in trees:
                self.assertEqual(len(t.nodes), 6)
                for n in t.nodes:
                    if n.parent is not None:  # forks only at "\n\n" segment boundaries
                        self.assertIn(n.offset, t.boundaries(n.parent))
                check_consistency(self, t)

    def test_p1_per_round_batched(self):
        g, trees = fake_trees()
        rngs = [random.Random(i) for i in range(len(trees))]
        M.run_p1(trees, g, rngs, budget=7, max_response=3000, max_model_len=4096, per_round=3, min_suffix=4)
        self.assertTrue(all(len(t.nodes) == 7 for t in trees))
        self.assertEqual(max(n.round for n in trees[0].nodes), 2)  # 1 + 3 + 3

    def test_p2_variants(self):
        ver = Verifier("string")
        for variant in M.P2_VARIANTS:
            for cands in ("all", "branch_children"):
                g, trees = fake_trees()
                rngs = [random.Random(i) for i in range(len(trees))]
                M.run_p1(trees, g, rngs, budget=6, max_response=3000, max_model_len=4096, min_suffix=4)
                M.score(trees, g, ver)
                for t in trees:
                    dist = M.state_distribution(SegTree(t), variant, 0.5, cands)
                    self.assertAlmostEqual(sum(dist.values()), 1.0)
                M.run_p2(trees, g, ver, rngs, k=3, variant=variant, max_response=3000,
                         max_model_len=4096, candidates=cands)
                for t in trees:
                    self.assertEqual(len(t.nodes), 9)
                    self.assertTrue(all(n.reward is not None for n in t.nodes))
                    check_consistency(self, t)

    def test_delta_softmax_mass_on_zero_nodes(self):
        """Documents the concern: with candidates='all', delta==0 nodes soak up mass."""
        g, trees = fake_trees(1, seed=3)
        M.run_p1(trees, g, [random.Random(0)], budget=8, max_response=3000, max_model_len=4096, min_suffix=4)
        M.score(trees, g, Verifier("string"))
        seg = SegTree(trees[0])
        vs = [v for v in seg.nodes if not v.is_root and seg.before_boxed(v.parent)]
        zero = sum(seg.delta(v) == 0 for v in vs)
        self.assertGreater(zero / len(vs), 0.5)


class TestOracleAndCli(unittest.TestCase):
    def test_oracle_records(self):
        ver = Verifier("string")
        g, trees = fake_trees(2)
        rngs = [random.Random(i) for i in range(2)]
        M.run_p1(trees, g, rngs, budget=5, max_response=3000, max_model_len=4096, min_suffix=4)
        M.score(trees, g, ver)
        states = [candidate_states(SegTree(t), [("rand", 0.5, "all"), ("spread", 0.5, "all")], 5)[0]
                  for t in trees]
        recs = run_oracle(trees, states, g, ver, R=6, K=3, M=2, max_response=3000, max_model_len=4096)
        self.assertEqual(len(recs), sum(len(s) for s in states))
        for r in recs:
            self.assertAlmostEqual(r["var_total"], r["p_hat"] * (1 - r["p_hat"]))
            self.assertEqual(r["n_rollouts"], 6)

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            data = f"{d}/toy.jsonl"
            with open(data, "w") as f:
                for i in range(3):
                    f.write(json.dumps({"id": f"toy{i}", "problem": f"toy problem {i}", "answer": "7"}) + "\n")
            common = ["--backend", "fake", "--chunk", "2"]
            main(["gen", "--method", "b0", "--n", "4", "--data", data, "--limit", "3",
                  "--out", f"{d}/b0.jsonl"] + common)
            main(["gen", "--method", "b2", "--M", "2", "--N", "1", "--L", "1", "--T", "2", "--data", data,
                  "--limit", "3", "--out", f"{d}/b2.jsonl"] + common)
            main(["gen", "--method", "p1", "--budget", "5", "--min_suffix", "4", "--data", data,
                  "--limit", "3", "--out", f"{d}/p1.jsonl"] + common)
            main(["gen", "--method", "p1", "--budget", "5", "--min_suffix", "4", "--data", data,
                  "--limit", "3", "--out", f"{d}/p1.jsonl"] + common)  # resume: nothing new
            with open(f"{d}/p1.jsonl") as f:
                self.assertEqual(len(f.readlines()), 3)
            main(["p2", "--from", f"{d}/p1.jsonl", "--variant", "absdelta", "--k", "2",
                  "--out", f"{d}/p2.jsonl"] + common)
            main(["oracle", "--from", f"{d}/p1.jsonl", "--R", "4", "--K", "2", "--Mq", "2",
                  "--max_states", "4", "--out", f"{d}/oracle.jsonl"] + common)
            main(["summary", f"{d}/b0.jsonl", f"{d}/b2.jsonl", f"{d}/p1.jsonl", f"{d}/p2.jsonl"])
            main(["report", "--p1", f"{d}/p1.jsonl", "--oracle", f"{d}/oracle.jsonl",
                  "--trees", f"{d}/p2.jsonl"])


class TestMisc(unittest.TestCase):
    def test_verifier(self):
        v = Verifier("string")
        self.assertEqual(extract_boxed("so \\boxed{\\frac{1}{2}} done"), "\\frac{1}{2}")
        self.assertEqual(v("\\boxed{\\dfrac{1}{2}}", "\\frac{1}{2}"), 1.0)
        self.assertEqual(v("\\boxed{142}", "142.0"), 1.0)
        self.assertEqual(v("no box", "1"), 0.0)

    def test_data_loaders(self):
        for name in ("MATH500.jsonl", "olympiad_bench.jsonl"):
            ps = load_problems(os.path.join(REPO, "datasets", "eval", name), limit=3)
            self.assertEqual(len(ps), 3)
            self.assertFalse(ps[0].answer.startswith("["))

    def test_pass_at_k(self):
        from tree_alloc.task2_report import pass_at_k

        self.assertAlmostEqual(pass_at_k(4, 1, 2), 0.5)  # 1 - C(3,2)/C(4,2)
        self.assertEqual(pass_at_k(4, 0, 4), 0.0)
        self.assertEqual(pass_at_k(4, 3, 2), 1.0)
        # unbiased: averaging first-k success over all k-subsets equals pass_at_k
        import itertools

        rs = [1, 0, 0, 1, 0]
        subsets = list(itertools.combinations(rs, 3))
        self.assertAlmostEqual(sum(max(s) for s in subsets) / len(subsets), pass_at_k(5, 2, 3))

    def test_stats(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        b = paired_bootstrap({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 0.0, "b": 1.0, "c": 2.0})
        self.assertAlmostEqual(b["diff"], 1.0)

    def test_tree_metrics(self):
        g, trees = fake_trees(1)
        M.run_iid(trees, g, 4, 3000, 4096)
        M.score(trees, g, Verifier("string"))
        m = tree_metrics(trees[0])
        self.assertEqual(m["n_leaves"], 4)
        self.assertTrue(0.0 <= m["delta_zero_frac"] <= 1.0)
        self.assertEqual(m["fork_rel_pos"], [])


if __name__ == "__main__":
    unittest.main()
