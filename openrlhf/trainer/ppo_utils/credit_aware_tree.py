"""Novelty-aware, credit-guided rollout allocation (Phase II of our proposal, Sec. 4.3).

Phase I is TreeRL's EPTree, unchanged: EntropyGuidedChainLocalManager.entropy_guided_chain builds the
n - k leaves of T0 and grades them. Phase II then spends k more rollouts, one at a time.

States are token positions, as in TreeRL's forking: state (node, p) is the prompt plus the response up
to, not including, token p of TreeNode `node`; forking there generates a new token p. Eligible states
are the ones TreeRL may fork at (node.mask[p] False), except the prompt itself (p = 0 of a first chain).

  L(u)       = leaves whose response passes through state u
  H(l, u)    = -(1 / L) * sum_t log p(x_t | c_u, x_<t)     over the L tokens of leaf l after u
  D(u)       = NOVELTY_SCORES[novelty_score]([H(l, u) for l in L(u)])   # default exp(mean): perplexity
  V_filtered = top rho fraction of T0's eligible states by D(u), fixed for the whole phase
  N_child    = children of V_filtered, recomputed every step so new branches join
  for t = 1..k:
      v* = argmax over N_child of ALLOC_SCORES[alloc_score](v, t)      # default UCB on |delta|
      generate one continuation from parent(v*), grade it, add it to the tree

A child of (node, p) is the state one token deeper: (node, p + 1), continuing node's own tokens, and
(c, 1) for every generation c that forks from node at p.

To add an alternative novelty or allocation score, register a function below; the CLI choices
(--novelty_score, --alloc_score in train_reinforce_ray.py) are read from these dicts.
"""
import math
import random
from bisect import bisect_left
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from openrlhf.trainer.ppo_utils.entropy_chain_local_manager import EntropyGuidedChainLocalManager
    from openrlhf.trainer.ppo_utils.evaluation import query_local_vllm_ids_with_logprobs
    from openrlhf.trainer.ppo_utils.tree_node import TreeNode
except:  # imported as a plain module (ppo_utils on sys.path), like TreeRL's own managers
    from entropy_chain_local_manager import EntropyGuidedChainLocalManager
    from evaluation import query_local_vllm_ids_with_logprobs
    from tree_node import TreeNode


@dataclass
class Candidate:
    """What an allocation score may look at for a state v in N_child."""
    value: float  # V(v): mean reward of the leaves through v
    parent_value: float  # V(parent(v))
    num_leaves: int  # |L(v)|
    novelty: Optional[float]  # D(v), None when v is not an eligible state of T0

    @property
    def delta(self) -> float:
        return self.value - self.parent_value


# novelty: list of H(l, u) over the leaves through u -> D(u)
NOVELTY_SCORES: Dict[str, Callable[[List[float]], float]] = {
    "perplexity": lambda hs: math.exp(sum(hs) / len(hs)),  # Eq. 4
}

# allocation: (candidate, t in 1..k) -> score; the argmax is expanded
ALLOC_SCORES: Dict[str, Callable[[Candidate, int], float]] = {
    "ucb": lambda c, t: abs(c.delta) + math.sqrt(2 * math.log(t) / c.num_leaves),  # Eqs. 7, 9
}

State = Tuple[TreeNode, int]


def _subtree_leaves(node: TreeNode) -> List[TreeNode]:
    """Every TreeNode is one leaf (its own response); a subtree's leaves are its nodes."""
    out = [node]
    for child in node.child_nodes:
        out += _subtree_leaves(child)
    return out


class CreditAwareChainLocalManager(EntropyGuidedChainLocalManager):
    """EPTree (Phase I) followed by Phase II. Phase II runs inside evaluate_trees, right after T0's
    leaves are graded, so the rest of TreeRL's pipeline (values, path selection) sees all n leaves."""

    def evaluate_trees(self, problem_str: str, answer_str: str, args: Dict[str, Any]) -> List[float]:
        pass_k_result = super().evaluate_trees(problem_str, answer_str, args)
        if args.get("phase2_k", 0) <= 0:
            return pass_k_result
        self.counterfactual_expand(problem_str, args)
        return [node.binary_score for tree_list in self.tree_lists for node in tree_list]

    # ---- tree statistics ----------------------------------------------
    def _eligible(self, node: TreeNode, p: int) -> bool:
        return not node.mask[p] and not (node.parent_node is None and p == 0)

    def _index(self) -> Dict[int, tuple]:
        """Per TreeNode: its children's split points (sorted) and suffix totals of their subtrees'
        (reward sum, leaf count), so the leaves through any state are counted with one bisect."""
        idx = {}

        def visit(node: TreeNode) -> Tuple[float, int]:
            kids = sorted(node.child_nodes, key=lambda c: c.parent_node_split_idx)
            sub = [visit(c) for c in kids]
            suf_r, suf_n = [0.0] * (len(kids) + 1), [0] * (len(kids) + 1)
            for i in range(len(kids) - 1, -1, -1):
                suf_r[i], suf_n[i] = suf_r[i + 1] + sub[i][0], suf_n[i + 1] + sub[i][1]
            own = float(node.binary_score or 0)
            idx[id(node)] = ([c.parent_node_split_idx for c in kids], suf_r, suf_n, own)
            return own + suf_r[0], 1 + suf_n[0]

        for tree_list in self.tree_lists:
            visit(tree_list[0])
        return idx

    @staticmethod
    def _stats(idx: Dict[int, tuple], node: TreeNode, p: int) -> Tuple[float, int]:
        """(reward sum, leaf count) of the leaves through (node, p): node itself and the subtrees of
        the children that fork at p or later (they share node's tokens 0..p-1)."""
        splits, suf_r, suf_n, own = idx[id(node)]
        i = bisect_left(splits, p)
        return own + suf_r[i], 1 + suf_n[i]

    @staticmethod
    def _children(node: TreeNode, p: int) -> List[Tuple[State, str]]:
        kids = [((node, p + 1), "continue")]
        kids += [((c, 1), "branch") for c in node.child_nodes if c.parent_node_split_idx == p]
        return kids

    def _novelty(self, novelty_fn: Callable[[List[float]], float]) -> Dict[tuple, Tuple[State, float]]:
        """D(u) for every eligible state of T0 (Eqs. 2-4): key -> ((node, p), D).

        Every leaf through state u shares u's prefix, so with C(u) = log-prob sum of that prefix,
        H(l, u) = -(total_l - C(u)) / (end_l - cut): one number per state and two per leaf, instead of
        a copy of each leaf's full log-prob list."""
        nodes = [node for tree_list in self.tree_lists for node in tree_list]
        own = {id(n): [0.0] + list(accumulate(n.log_prob_list)) for n in nodes}  # prefix sums of own tokens
        base: Dict[int, float] = {}  # log-prob sum of the prefix a node inherits from its ancestors

        def base_of(node: TreeNode) -> float:
            if id(node) not in base:
                parent = node.parent_node
                base[id(node)] = 0.0 if parent is None else \
                    base_of(parent) + own[id(parent)][node.parent_node_split_idx]
            return base[id(node)]

        total = {id(n): base_of(n) + own[id(n)][-1] for n in nodes}  # leaf's full-response log-prob
        end = {id(n): len(n.aggregate_token_ids) + len(n.log_prob_list) for n in nodes}  # its length

        novelty = {}
        for node in nodes:
            start, b, oc = len(node.aggregate_token_ids), base_of(node), own[id(node)]
            kids = sorted(node.child_nodes, key=lambda c: c.parent_node_split_idx, reverse=True)
            through, j = [node], 0  # leaves through (node, p), growing as p moves left
            for p in range(len(node.token_id_list) - 1, -1, -1):
                while j < len(kids) and kids[j].parent_node_split_idx >= p:
                    through += _subtree_leaves(kids[j])
                    j += 1
                if not self._eligible(node, p):
                    continue
                cut, c_u = start + p, b + oc[p]
                hs = [-(total[id(leaf)] - c_u) / (end[id(leaf)] - cut) for leaf in through if end[id(leaf)] > cut]
                if hs:
                    novelty[(id(node), p)] = ((node, p), novelty_fn(hs))
        return novelty

    # ---- Phase II -----------------------------------------------------
    def counterfactual_expand(self, problem_str: str, args: Dict[str, Any]) -> None:
        k, rho = args["phase2_k"], args.get("phase2_rho", 1.0)
        novelty_fn = NOVELTY_SCORES[args.get("novelty_score", "perplexity")]
        alloc_fn = ALLOC_SCORES[args.get("alloc_score", "ucb")]

        novelty = self._novelty(novelty_fn)
        if not novelty:
            raise ValueError("Phase II found no eligible fork position in T0")
        # Eq. 5: keep the top rho fraction (at least one state); fixed for the whole phase
        ranked = sorted(novelty, key=lambda key: novelty[key][1], reverse=True)
        filtered = [novelty[key][0] for key in ranked[:max(1, math.ceil(rho * len(ranked)))]]

        self.phase2_log = []
        for t in range(1, k + 1):
            idx = self._index()
            best, choices = -math.inf, []
            for node, p in filtered:
                r_u, n_u = self._stats(idx, node, p)
                for (child, q), kind in self._children(node, p):  # Eq. 8
                    r, n = self._stats(idx, child, q)
                    entry = novelty.get((id(child), q))
                    c = Candidate(value=r / n, parent_value=r_u / n_u, num_leaves=n,
                                  novelty=entry[1] if entry else None)
                    score = alloc_fn(c, t)
                    if score > best:
                        best, choices = score, [(node, p, c, kind)]
                    elif score == best:
                        choices.append((node, p, c, kind))
            node, p, c, kind = random.choice(choices)  # Eq. 10, random tie-break

            # expand parent(v*) = the filtered state (node, p): a new branch forking at token p
            new = self._generate_from(node, p, args)
            new.binary_score, new.score = self.evaluate_node(args, problem_str, new)
            new.phase2 = True
            self.phase2_log.append({
                "t": t, "score": best, "ties": len(choices), "child": kind, "delta": c.delta,
                "num_leaves": c.num_leaves, "novelty": novelty[(id(node), p)][1],
                "fork_node_idx": node.node_idx, "tree_idx": node.tree_idx, "split_idx": p,
                "reward": new.binary_score,
            })

    def _generate_from(self, node: TreeNode, split_idx: int, args: Dict[str, Any]) -> TreeNode:
        """One continuation from node's token split_idx, attached as in EPTree's fork round."""
        prompt_ids = self.paths["init_prompt_ids_with_template"] + node.get_prefix_ids(split_idx)
        for _ in range(4):
            result = query_local_vllm_ids_with_logprobs(
                [prompt_ids],
                llm=self.llm,
                skip_special_tokens=False,
                max_tokens=args["generate_max_len"],
                stops=self.eos_tokens_set,
                temperature=self.args["temperature"],
                top_p=self.args["top_p"],
            )
            if result is not None and result[0] is not None:
                break
        else:
            raise RuntimeError("Phase II generation failed 4 times")
        content_token_ids, _, finish_reasons, _, log_probs = result
        new = TreeNode(
            tree_idx=node.tree_idx,
            node_idx=len(self.tree_lists[node.tree_idx]),
            token_id_list=content_token_ids[0],
            decode_fn=self.decode_fn,
            log_prob_list=log_probs[0],
            is_end=True,
            parent_node=node,
            parent_node_idx=node.node_idx,
            parent_node_split_idx=split_idx,
            finish_reason=finish_reasons[0],
        )
        node.add_child(new, split_idx)
        self.tree_lists[node.tree_idx].append(new)
        return new
