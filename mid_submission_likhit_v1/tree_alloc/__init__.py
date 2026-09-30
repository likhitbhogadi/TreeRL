"""Tier-1 (inference-only) harness for credit-aware counterfactual rollout allocation.

Standalone: does not import `openrlhf`. vLLM and math_verify are imported lazily,
so the tree / statistics / method logic runs (and is unit-tested) without a GPU.
"""
