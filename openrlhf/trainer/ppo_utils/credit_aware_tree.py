"""Novelty-aware, credit-guided rollout allocation (Phase II of our proposal, Sec. 4.3).

Phase I is TreeRL's EPTree, unchanged: EntropyGuidedChainLocalManager.entropy_guided_chain builds the
n - k leaves of T0 and grades them. Phase II then spends k more rollouts, one at a time.

States are token positions, as in TreeRL's forking: state u is a context c_u = prompt + a response
prefix; forking at u generates a new token there. Positions with identical prefixes (e.g. the opening
tokens shared by several chains) are the same context, so the observed responses are kept in a trie
with one trie node per distinct context. A state is eligible if TreeRL may fork at one of its positions
(node.mask[p] False), except the prompt itself.

  E_u^h    = distinct h-token prefixes of the continuations observed from u (incl. truncated ones;
             a continuation that ends within h tokens counts as itself)
  N_h(u)   = 1 - sum over Y in E_u^h of prod_t p(y_t | c_u, y_<t)
           = P(a new rollout from u differs from every observed continuation within h tokens)
  V(v)     = mean reward of the leaves through v
  C(v)     = |V(v) - V(parent(v))|
  P        = eligible states,  V_cand = their children (one token deeper)
  N~, C~   = min-max normalisation of N_h over P and of C over V_cand (0 if the range is 0)
  J(v)     = lambda * N~(parent(v)) + (1 - lambda) * C~(v)
  for t = 1..k:  v* = argmax J (random tie-break); generate one continuation from parent(v*),
                 grade it, add it to the tree; recompute everything on the updated tree

N_h follows the recursion over the distinct next tokens a of u (children of u in the trie):
  C_d(u) = 1                                  if d = 0 or an observed response ends at u
  C_d(u) = sum_a p(a | c_u) * C_{d-1}(child_a)   otherwise;     N_h(u) = 1 - C_h(u)
"""
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from openrlhf.trainer.ppo_utils.entropy_chain_local_manager import EntropyGuidedChainLocalManager
    from openrlhf.trainer.ppo_utils.evaluation import query_local_vllm_ids_with_logprobs
    from openrlhf.trainer.ppo_utils.tree_node import TreeNode
except:  # imported as a plain module (ppo_utils on sys.path), like TreeRL's own managers
    from entropy_chain_local_manager import EntropyGuidedChainLocalManager
    from evaluation import query_local_vllm_ids_with_logprobs
    from tree_node import TreeNode


class ContextTrie:
    """Observed responses merged on identical token prefixes. Trie node 0 is the prompt; a trie node's
    id is larger than its parent's, so iterating ids downwards visits children before parents."""

    def __init__(self):
        self.parent: List[int] = [-1]
        self.logp: List[float] = [0.0]  # log p of the token leading into the trie node
        self.kids: List[Dict[int, int]] = [{}]  # token id -> trie node
        self.ends: List[List[TreeNode]] = [[]]  # responses that end at the trie node
        self.fork: List[Optional[Tuple[TreeNode, int]]] = [None]  # an eligible (TreeNode, p) here
        self.pos: Dict[int, List[int]] = {}  # id(TreeNode) -> trie nodes of its positions 0..len

    def add(self, node: TreeNode, eligible: Callable[[TreeNode, int], bool]) -> None:
        """Insert node's own tokens below the position it forks from (its parent must be added first)."""
        u = 0 if node.parent_node is None else self.pos[id(node.parent_node)][node.parent_node_split_idx]
        ids = [u]
        for p, (tok, lp) in enumerate(zip(node.token_id_list, node.log_prob_list)):
            if self.fork[u] is None and eligible(node, p):
                self.fork[u] = (node, p)
            v = self.kids[u].get(tok)
            if v is None:  # the first observation of this token in context u keeps its log-prob
                v = len(self.parent)
                self.parent.append(u)
                self.logp.append(lp)
                self.kids.append({})
                self.ends.append([])
                self.fork.append(None)
                self.kids[u][tok] = v
            ids.append(v)
            u = v
        self.ends[u].append(node)
        self.pos[id(node)] = ids

    def stats(self, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(V, leaf count, N_h) for every trie node, in one children-before-parents pass."""
        n = len(self.parent)
        reward, count = np.zeros(n), np.zeros(n)
        cover = np.zeros((n, h + 1))  # cover[u, d] = C_d(u)
        cover[:, 0] = 1.0
        for u, ends in enumerate(self.ends):
            if ends:
                reward[u] = sum(float(leaf.binary_score or 0) for leaf in ends)
                count[u] = len(ends)
                cover[u] = 1.0
        prob = np.exp(self.logp)
        for v in range(n - 1, 0, -1):
            u = self.parent[v]
            reward[u] += reward[v]
            count[u] += count[v]
            if not self.ends[u]:
                cover[u, 1:] += prob[v] * cover[v, :-1]
        return reward / count, count, 1.0 - cover[:, h]


def _min_max(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class CreditAwareChainLocalManager(EntropyGuidedChainLocalManager):
    """EPTree (Phase I) followed by Phase II. Phase II runs inside evaluate_trees, right after T0's
    leaves are graded, so the rest of TreeRL's pipeline (values, path selection) sees all n leaves."""

    def evaluate_trees(self, problem_str: str, answer_str: str, args: Dict[str, Any]) -> List[float]:
        pass_k_result = super().evaluate_trees(problem_str, answer_str, args)
        if args.get("phase2_k", 0) <= 0:
            return pass_k_result
        self.counterfactual_expand(problem_str, args)
        return [node.binary_score for tree_list in self.tree_lists for node in tree_list]

    def _eligible(self, node: TreeNode, p: int) -> bool:
        return not node.mask[p] and not (node.parent_node is None and p == 0)

    def counterfactual_expand(self, problem_str: str, args: Dict[str, Any]) -> None:
        k, lam, h = args["phase2_k"], args.get("phase2_lambda", 0.5), args.get("phase2_horizon", 20)

        trie = ContextTrie()
        for tree_list in self.tree_lists:  # parents precede children in each list
            for node in tree_list:
                trie.add(node, self._eligible)

        self.phase2_log = []
        for t in range(1, k + 1):
            value, count, novelty = trie.stats(h)
            parents = np.array([u for u, f in enumerate(trie.fork) if f is not None])
            if len(parents) == 0:
                raise ValueError("Phase II found no eligible fork position")
            cand_parent, cand = zip(*[(i, v) for i, u in enumerate(parents) for v in trie.kids[u].values()])
            cand_parent, cand = np.array(cand_parent), np.array(cand)

            n_norm = _min_max(novelty[parents])
            credit = np.abs(value[cand] - value[parents[cand_parent]])
            c_norm = _min_max(credit)
            score = lam * n_norm[cand_parent] + (1 - lam) * c_norm
            ties = np.flatnonzero(score == score.max())
            j = random.choice(ties)
            u, v = parents[cand_parent[j]], cand[j]

            # expand parent(v*) = u: a new branch forking at one of u's eligible positions
            node, p = trie.fork[u]
            new = self._generate_from(node, p, args)
            new.binary_score, new.score = self.evaluate_node(args, problem_str, new)
            new.phase2 = True
            trie.add(new, self._eligible)
            self.phase2_log.append({
                "t": t, "J": float(score[j]), "ties": len(ties),
                "novelty": float(novelty[u]), "novelty_norm": float(n_norm[cand_parent[j]]),
                "credit": float(credit[j]), "credit_norm": float(c_norm[j]),
                "delta": float(value[v] - value[u]), "num_leaves": int(count[v]),
                "num_states": len(parents), "num_candidates": len(cand),
                "tree_idx": node.tree_idx, "fork_node_idx": node.node_idx, "split_idx": p,
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
