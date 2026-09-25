"""Generation backends. All forking is done on token ids (prompt_ids + response[:p]);
prefixes are never decoded and re-tokenized, and no EOS / template tokens are appended.
"""
from __future__ import annotations

import functools
import hashlib
import os
import random
from typing import List, Optional, Sequence

from .tree import Generation

QWEN_MATH_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


class Generator:
    """Interface used by the methods."""

    def encode_prompt(self, question: str) -> List[int]:
        raise NotImplementedError

    def token_text(self, token_id: int) -> str:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def generate(self, prompts: List[List[int]], max_tokens: List[int],
                 stop: Optional[List[str]] = None) -> List[Generation]:
        """One sample per prompt; returns sampled token ids and their logprobs."""
        raise NotImplementedError


class VLLMGenerator(Generator):
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

    def generate(self, prompts, max_tokens, stop=None):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        assert len(prompts) == len(max_tokens), (len(prompts), len(max_tokens))
        params = [
            SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=m,
                           logprobs=1, stop=stop, include_stop_str_in_output=bool(stop))
            for m in max_tokens
        ]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params,
                                 use_tqdm=False)
        gens = []
        for o in outs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            lps = [c.logprobs[i][t].logprob for i, t in enumerate(ids)]
            gens.append(Generation(ids, lps, "length" if c.finish_reason == "length" else "stop"))
        return gens


class FakeGenerator(Generator):
    """Deterministic toy LM for tests: emits "\\n\\n"-separated steps and a boxed answer.

    Each sample is seeded by (prompt, call counter) so runs are reproducible. The
    answer is "7" with probability `p_correct` if the context so far contains an even
    number of " x" tokens, else 1 - p_correct -- so earlier steps are consequential
    and the oracle has real decision-level signal to find.
    """

    VOCAB = ["Let", " x", " =", " 2", " so", " we", " get", ".", "\n\n", "\\", "boxed", "{", "7", "3", "}", "\n"]

    def __init__(self, seed: int = 0, p_correct: float = 0.8, min_steps: int = 2, max_steps: int = 5):
        self.seed, self.p_correct = seed, p_correct
        self.min_steps, self.max_steps = min_steps, max_steps
        self.calls = 0
        self.id_of = {s: i for i, s in enumerate(self.VOCAB)}

    def encode_prompt(self, question: str) -> List[int]:
        return [100 + (ord(c) % 50) for c in question[:20]]

    def token_text(self, token_id: int) -> str:
        return self.VOCAB[token_id] if 0 <= token_id < len(self.VOCAB) else f"<p{token_id}>"

    def decode(self, ids):
        return "".join(self.token_text(i) for i in ids)

    def _sample(self, prompt: List[int], max_tok: int, stop: Optional[List[str]]) -> Generation:
        h = hashlib.sha256(f"{self.seed}|{self.calls}|{prompt}".encode()).hexdigest()
        rng = random.Random(int(h[:16], 16))
        self.calls += 1
        words = ["Let", " x", " =", " 2", " so", " we", " get"]
        ids: List[int] = []
        for _ in range(rng.randint(self.min_steps, self.max_steps)):
            ids += [self.id_of[w] for w in rng.choices(words, k=rng.randint(3, 8))]
            ids += [self.id_of["."], self.id_of["\n\n"]]
            if stop and "\n\n" in stop:
                break
        else:
            even = (prompt + ids).count(self.id_of[" x"]) % 2 == 0
            p = self.p_correct if even else 1 - self.p_correct
            ans = "7" if rng.random() < p else "3"
            ids += [self.id_of[s] for s in ["\\", "boxed", "{", ans, "}"]]
        finish = "stop"
        if len(ids) > max_tok:
            ids, finish = ids[:max_tok], "length"
        lps = [-rng.expovariate(1.5) for _ in ids]
        return Generation(ids, lps, finish)

    def generate(self, prompts, max_tokens, stop=None):
        return [self._sample(p, m, stop) for p, m in zip(prompts, max_tokens)]
