"""Run from mid_submission/:  python3 -m unittest discover -s tests -v   (no GPU needed)"""
import itertools
import json
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_treerl import score, to_tree  # noqa: E402
from tree_alloc.data import load_problems  # noqa: E402
from tree_alloc.metrics import paired_bootstrap, tree_metrics  # noqa: E402
from tree_alloc.segments import SegTree  # noqa: E402
from tree_alloc.task2_report import pass_at_k  # noqa: E402
from tree_alloc.tree import Generation, Tree, annotate  # noqa: E402
from tree_alloc.verify import Verifier, extract_boxed  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VOCAB = ["Let", " x", " =", " 2", " so", " we", " get", ".", "\n\n", "\\", "boxed", "{", "7", "3", "}"]
ID = {w: i for i, w in enumerate(VOCAB)}


def text(i):
    return VOCAB[i]


def toks(*words):
    return [ID[w] for w in words]


def gen(ids, lp=-0.5, finish="stop"):
    return Generation(list(ids), [lp] * len(ids), finish)


def check_consistency(tc, t):
    """Each branch continues exactly its parent's prefix at the fork point."""
    for n in t.nodes:
        ids, _ = t.response(n.id)
        tc.assertEqual(t.prefix_ids(n.parent, n.fork_idx), ids[:n.offset])


class TestAnnotate(unittest.TestCase):
    def test_boundaries_and_boxed(self):
        tx = {0: "a", 1: ".\n\n", 2: "b", 3: "\n", 4: "\n", 5: "c", 6: "\\", 7: "boxed", 8: "{"}
        seg_ends, boxed = annotate(list(range(9)), tx.get)
        self.assertEqual(seg_ends, [2, 5])  # after ".\n\n" and after the split "\n","\n"
        self.assertEqual(boxed, 6)  # start of "\boxed", not the token completing it

    def test_newline_run_is_one_boundary(self):
        tx = {0: "a", 1: "\n\n", 2: "\n", 3: "b"}
        self.assertEqual(annotate([0, 1, 2, 3], tx.get)[0], [3])


class TestTree(unittest.TestCase):
    def setUp(self):
        self.t = Tree("p", [101, 102], "7")
        s1 = toks("Let", " x", ".", "\n\n")
        s2 = toks(" so", " we", ".", "\n\n")
        ans = toks("\\", "boxed", "{", "7", "}")
        self.t.add(None, 0, gen(s1 + s2 + ans), text)  # node 0
        self.t.add(0, 4, gen(toks(" get", ".", "\n\n") + toks("\\", "boxed", "{", "3", "}")), text)  # node 1
        self.t.add(1, 3, gen(toks("\\", "boxed", "{", "7", "}")), text)  # node 2, inside node 1

    def test_prefix_and_boundaries(self):
        check_consistency(self, self.t)
        self.assertEqual(self.t.nodes[2].offset, 7)
        self.assertEqual(self.t.boundaries(2), [4, 7, 12])
        self.assertEqual(self.t.boxed_pos(0), 8)
        self.assertEqual(self.t.new_tokens, 13 + 8 + 5)

    def test_json_roundtrip(self):
        t2 = Tree.from_json(json.loads(json.dumps(self.t.to_json())))
        self.assertEqual(t2.response(2), self.t.response(2))

    def test_segments_and_values(self):
        for n, r in zip(self.t.nodes, [1.0, 0.0, 1.0]):
            n.reward = r
        seg = SegTree(self.t)
        for leaf in range(3):  # every leaf's segment path ends at its response length
            v = next(v for v in seg.nodes if leaf in v.ends)
            self.assertEqual(v.end_pos, len(self.t.response(leaf)[0]))
        root, first = seg.root, seg.root.children[0]
        self.assertEqual(len(root.children), 1)
        self.assertEqual(seg.value(first), seg.value(root))  # single child: same leaves
        self.assertEqual([len(u.children) for u in seg.branch_points()], [2, 2])
        self.assertAlmostEqual(seg.value(root), 2 / 3)

    def test_tree_metrics(self):
        for n in self.t.nodes:
            n.reward, n.answer = 1.0, "7"
        m = tree_metrics(self.t)
        self.assertEqual(m["n_leaves"], 3)
        self.assertEqual(m["pass_any"], 1.0)
        self.assertEqual(m["mixed"], 0.0)
        self.assertEqual(len(m["fork_rel_pos"]), 2)


class TestRunner(unittest.TestCase):
    """TreeRL TreeNode lists -> our Tree (run_treerl.to_tree), then re-grading."""

    def test_to_tree_and_score(self):
        def node(ids, parent=None, split=None, grade=1):
            return SimpleNamespace(token_id_list=ids, log_prob_list=[-0.5] * len(ids), finish_reason="stop",
                                   parent_node=parent, parent_node_split_idx=split, binary_score=grade)
        root = node(toks("Let", " x", ".", "\n\n", "\\", "boxed", "{", "7", "}"))
        child = node(toks(" so", "\\", "boxed", "{", "3", "}"), parent=root, split=4, grade=0)
        other = node(toks("\\", "boxed", "{", "7", "}"))
        problem = SimpleNamespace(id="q1", answer="7")
        t, grades = to_tree(problem, [101], [[root, child], [other]], text, {"method": "b2"})
        self.assertEqual([n.parent for n in t.nodes], [None, 0, None])
        self.assertEqual(t.nodes[1].offset, 4)
        self.assertEqual(grades, [1, 0, 1])
        check_consistency(self, t)
        score(t, lambda ids: "".join(text(i) for i in ids), Verifier("string"))
        self.assertEqual([n.reward for n in t.nodes], [1.0, 0.0, 1.0])
        self.assertEqual([n.answer for n in t.nodes], ["7", "3", "7"])


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
        self.assertAlmostEqual(pass_at_k(4, 1, 2), 0.5)  # 1 - C(3,2)/C(4,2)
        self.assertEqual(pass_at_k(4, 0, 4), 0.0)
        self.assertEqual(pass_at_k(4, 3, 2), 1.0)
        rs = [1, 0, 0, 1, 0]  # unbiased: average over all k-subsets equals pass_at_k
        subsets = list(itertools.combinations(rs, 3))
        self.assertAlmostEqual(sum(max(s) for s in subsets) / len(subsets), pass_at_k(5, 2, 3))

    def test_paired_bootstrap(self):
        b = paired_bootstrap({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 0.0, "b": 1.0, "c": 2.0})
        self.assertAlmostEqual(b["diff"], 1.0)


if __name__ == "__main__":
    unittest.main()
