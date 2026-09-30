"""vLLM model loading for the TreeRL runner: raw logprobs, chat-template prompts as token ids,
per-token decoding. Sampling itself is done by TreeRL's code (evaluation.query_local_vllm_ids_with_logprobs)."""
from __future__ import annotations

import functools
import os
from typing import List, Optional, Sequence

QWEN_MATH_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


class VLLMGenerator:
    def __init__(self, model: str, temperature: float = 1.0, top_p: float = 0.95,
                 max_model_len: int = 4096, seed: int = 0, gpu_memory_utilization: float = 0.9,
                 system_prompt: Optional[str] = QWEN_MATH_SYSTEM, logprobs_mode: str = "raw_logprobs"):
        # FlashInfer's sampler JIT-compiles with ninja + a CUDA toolkit matching torch; the
        # lab box's system nvcc is 12.0 vs torch cu129, so use vLLM's PyTorch sampler.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM  # lazy: only needed on the GPU box
        from transformers import AutoTokenizer

        kwargs = dict(model=model, enable_prefix_caching=True, max_model_len=max_model_len,
                      seed=seed, gpu_memory_utilization=gpu_memory_utilization)
        try:
            # Surprisal must be under pi_theta, not the temperature/top-p processed distribution.
            self.llm = LLM(**kwargs, logprobs_mode=logprobs_mode)
            self.logprobs_mode = logprobs_mode
        except TypeError:
            print("[gen] this vLLM has no `logprobs_mode`; check whether returned logprobs are "
                  "raw or processed for your version and state it in the report")
            self.llm = LLM(**kwargs)
            self.logprobs_mode = "vllm-default"
        self.tok = AutoTokenizer.from_pretrained(model)
        self.temperature, self.top_p = temperature, top_p
        self.max_model_len = max_model_len
        self.system_prompt = system_prompt

    def encode_prompt(self, question: str) -> List[int]:
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": question})
        out = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if hasattr(out, "keys"):  # transformers>=5 returns a BatchEncoding, not a list of ids
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        ids = [int(i) for i in out]
        assert ids, "empty prompt"
        return ids

    @functools.lru_cache(maxsize=None)
    def token_text(self, token_id: int) -> str:
        return self.tok.decode([token_id], skip_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)
