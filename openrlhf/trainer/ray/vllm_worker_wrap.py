import torch

from openrlhf.utils.logging import init_logger

logger = init_logger(__name__)


class WorkerWrap:
    """vLLM (>=0.10, V1 engine) worker extension, passed as `worker_extension_cls`: its methods become
    worker methods callable through `LLM.collective_rpc`.

    Replaces the old `vllm.worker.worker.Worker` subclass (that module is gone in vLLM V1) and its NCCL
    broadcast, which cannot work when the actor and vLLM share one GPU (NCCL rejects two ranks on the
    same device). The actor sends CUDA IPC handles of its own weight tensors instead: no copy, no NCCL.
    """

    def update_weights_cuda_ipc(self, handles):
        """handles: [(name, (rebuild_fn, rebuild_args))] from torch's `reduce_tensor(param)` in the actor."""
        weights = []
        for name, (rebuild, args) in handles:
            args = list(args)
            args[6] = self.device.index  # the producer's device index; same physical GPU here
            weights.append((name, rebuild(*args)))
        self.model_runner.get_model().load_weights(weights=weights)  # get_model() unwraps CUDA-graph wrappers
        torch.cuda.synchronize()
        del weights
