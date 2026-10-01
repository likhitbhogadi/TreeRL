import os

import ray

from openrlhf.utils.logging import init_logger

logger = init_logger(__name__)


@ray.remote
class LLMRayActor:
    """vLLM engine as a Ray actor, colocated on the actor's GPU (vLLM >= 0.10, V1 engine).

    The engine runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so `collective_rpc` passes the
    CUDA IPC handles to the worker as plain Python objects. Outside tree generation the engine sleeps
    (its GPU memory freed) so the logprob pass and the training step can use the memory.
    """

    def __init__(self, *args, keep_memory=False, **kwargs):
        # keep_memory: never sleep, so other users of a shared GPU cannot take vLLM's memory between rollouts
        # (a sleeping engine that cannot get its memory back fails on wake-up with CUDA out of memory)
        self.keep_memory = keep_memory
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # FlashInfer's sampler JIT-compiles with the system nvcc (12.0 on pkgpu2, too old for torch cu129);
        # use vLLM's PyTorch sampler, as mid_submission/tree_alloc/gen.py does
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        import vllm

        self.llm = vllm.LLM(*args, worker_extension_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
                            enable_sleep_mode=True, **kwargs)

    asleep = False

    def generate(self, prompts=None, sampling_params=None, prompt_token_ids=None, use_tqdm=False):
        # vLLM >= 0.10 dropped `prompt_token_ids=`; callers in ppo_utils still pass it
        if prompt_token_ids is not None:
            from vllm.inputs import TokensPrompt

            prompts = [TokensPrompt(prompt_token_ids=p) for p in prompt_token_ids]
        self.wake_up()
        return self.llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)

    def sleep(self):
        """Level 1: weights to CPU, KV cache freed. Waking is then always safe (weights come back), so
        the rollout can sleep the engine as soon as its trees are built."""
        if not self.asleep and not self.keep_memory:
            self.llm.sleep(level=1)
            self.asleep = True

    def wake_up(self):
        if self.asleep:
            self.llm.wake_up()
            self.asleep = False

    def update_weights_cuda_ipc(self, handles):
        self.llm.collective_rpc("update_weights_cuda_ipc", args=(handles,))
        self.llm.reset_prefix_cache()  # cached prefixes were computed with the old weights


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool = False,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    num_gpus: float = 1,
    keep_memory: bool = False,
):
    # ponytail: one colocated engine, TP=1 only (the single-GPU setup); multi-GPU needs NCCL weight sync back
    assert num_engines == 1 and tensor_parallel_size == 1, "only one colocated TP=1 vLLM engine is supported"
    return [
        LLMRayActor.options(num_cpus=1, num_gpus=num_gpus).remote(
            pretrain,
            keep_memory=keep_memory,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            seed=int(seed),
            max_model_len=max_model_len,
            enable_prefix_caching=enable_prefix_caching,
        )
    ]
