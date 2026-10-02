"""Check that the actor -> vLLM CUDA-IPC weight sync (reinforce_actor._broadcast_to_vllm) really changes
vLLM's weights: perturb an HF copy of the model, push it through sleep -> wake -> IPC update, and compare vLLM's sampled-token logprobs with HF's on the same tokens.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/check_vllm_weight_sync.py [model]
"""
import sys

import ray
import torch
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from openrlhf.trainer.ray.vllm_engine import create_vllm_engines

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
ray.init(num_gpus=1)
engine = create_vllm_engines(1, 1, model, 0, gpu_memory_utilization=0.12, max_model_len=1024, num_gpus=0.5)[0]
tok = AutoTokenizer.from_pretrained(model)
hf = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).cuda().eval()
prompt = tok.apply_chat_template([{"role": "user", "content": "What is 7 * 8?"}], add_generation_prompt=True,
                                 return_dict=False)


def gap():
    """max |vLLM logprob - HF logprob| over 32 greedy tokens"""
    out = ray.get(engine.generate.remote(prompt_token_ids=[prompt],
                                         sampling_params=SamplingParams(temperature=0, max_tokens=32, logprobs=0)))[0]
    ids = list(out.outputs[0].token_ids)
    v = torch.tensor([next(iter(d.values())).logprob for d in out.outputs[0].logprobs])
    with torch.no_grad():
        logits = hf(torch.tensor([prompt + ids], device="cuda")).logits[0, len(prompt) - 1:-1].float()
    h = logits.log_softmax(-1).gather(-1, torch.tensor(ids, device="cuda")[:, None])[:, 0].cpu()
    return (v - h).abs().max().item()


before = gap()
with torch.no_grad():
    for p in hf.parameters():
        p.add_(torch.randn_like(p) * p.float().std().to(p.dtype) * 0.05)
stale = gap()
ray.get(engine.sleep.remote())
torch.cuda.synchronize()
handles = [(n, reduce_tensor(p.data)) for n, p in hf.named_parameters()]
ray.get(engine.wake_up.remote())
ray.get(engine.update_weights_cuda_ipc.remote(handles))
synced = gap()
print(f"max |logprob gap| vs HF: same weights {before:.3f}, after perturbing HF only {stale:.3f}, after sync {synced:.3f}")
assert before < 0.1 and synced < 0.1 < stale, "weight sync failed"  # bf16 baseline gap is ~0.04
print("weight sync OK")
