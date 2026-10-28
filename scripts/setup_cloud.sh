#!/bin/bash
# One-time setup of a cloud GPU instance (JarvisLabs) for the RL runs (mid_submission/RERUN_PAPER.md, section 7):
#   cd /home/TreeRL && bash scripts/setup_cloud.sh
# Installs pkgpu2's package versions into $BASE/venv (vLLM 0.30.0 and torch 2.13.0 built for CUDA 12.9,
# transformers 5.17.0, DeepSpeed 0.19.7, Ray 2.58.0), downloads the model, compiles DeepSpeed's Adam, and writes
# $BASE/treerl_env.sh with this machine's settings. Everything lives under $BASE (default /home, which JarvisLabs
# keeps while the instance is paused), so a paused instance needs no second setup. Safe to re-run.
set -eo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-/home}
export UV_CACHE_DIR=$BASE/.uv-cache HF_HOME=$BASE/hf TORCH_EXTENSIONS_DIR=$BASE/torch_extensions
mkdir -p $BASE/ckpt

echo "== GPU and CUDA"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true
if command -v nvcc > /dev/null; then nvcc --version | tail -1; else echo "no nvcc on PATH: DeepSpeed's Adam may not compile"; fi
# DeepSpeed compiles its Adam with nvcc for this GPU. Blackwell (compute capability 12.x) needs CUDA >= 12.8: install
# the CUDA 12.9 compiler and headers (torch's CUDA version) from NVIDIA's apt repository if it is missing. They go to
# /usr/local/cuda-12.9, outside $BASE, so a resumed instance may need this step again (re-running setup does it).
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | cut -d. -f1)
nvcc_ver=$(nvcc --version 2> /dev/null | grep -o "release [0-9.]*" | awk '{print $2}')
if [ "$cap" -ge 12 ] && [ ! -x /usr/local/cuda-12.9/bin/nvcc ] && [ "$(printf '%s\n12.8\n' "${nvcc_ver:-0}" | sort -V | head -1)" != 12.8 ]; then
  echo "Blackwell GPU with nvcc ${nvcc_ver:-none}: installing the CUDA 12.9 compiler"
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      cuda-minimal-build-12-9 libcublas-dev-12-9 libcusparse-dev-12-9 libcusolver-dev-12-9 libcurand-dev-12-9 \
      cuda-nvrtc-dev-12-9 > $BASE/ckpt/cuda129_install.log 2>&1 || echo "CUDA 12.9 install failed (see $BASE/ckpt/cuda129_install.log)"
fi
CUDA_ENV=""
if [ -x /usr/local/cuda-12.9/bin/nvcc ]; then  # build DeepSpeed's ops with torch's CUDA version
  export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH
  CUDA_ENV="export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:\$PATH"
  echo "DeepSpeed ops will be built with $(nvcc --version | tail -1)"
fi

echo "== Python environment in $BASE/venv"
command -v uv > /dev/null || pip install -q uv
[ -x $BASE/venv/bin/python ] || uv venv --python 3.12 $BASE/venv
source $BASE/venv/bin/activate
# CUDA 12.9 builds, as on pkgpu2 (mid_submission_likhit/tier1_implementation.md); the default PyPI builds need CUDA 13
uv pip install "vllm==0.30.0+cu129" --extra-index-url https://wheels.vllm.ai/0.30.0/cu129 --torch-backend=cu129 \
    --index-strategy unsafe-best-match
uv pip install --torch-backend=cu129 transformers==5.17.0 deepspeed==0.19.7 ray==2.58.0 datasets==5.0.1 \
    accelerate==1.15.0 math-verify levenshtein multiprocess ipython matplotlib openai requests pyyaml
uv pip freeze > $BASE/ckpt/env_freeze.txt  # the exact versions, kept with the results

echo "== model"
python -c "from huggingface_hub import snapshot_download as d; print(d('Qwen/Qwen2.5-Math-1.5B-Instruct'))"

echo "== checks"
python - <<'EOF'
import torch, vllm, deepspeed, transformers, ray
assert torch.cuda.is_available(), "torch cannot use the GPU: is the driver older than CUDA 12.9?"
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)} | vllm {vllm.__version__} | "
      f"deepspeed {deepspeed.__version__} | transformers {transformers.__version__} | ray {ray.__version__}")
EOF
# DeepSpeed compiles its Adam on first use: GPU Adam for ADAM_OFFLOAD= (80 GB cards), CPU Adam for ADAM_OFFLOAD=1
GPU_ADAM=ok CPU_ADAM=ok
python -c "import torch; from deepspeed.ops.adam import FusedAdam as A; p = torch.nn.Parameter(torch.zeros(4, device='cuda')); \
o = A([p], lr=0.1); p.grad = torch.ones_like(p); o.step(); assert p[0].item() < 0" > $BASE/ckpt/adam_gpu_build.log 2>&1 \
    || GPU_ADAM=failed
python -c "import torch; from deepspeed.ops.adam import DeepSpeedCPUAdam as A; p = torch.nn.Parameter(torch.zeros(4)); \
o = A([p], lr=0.1); p.grad = torch.ones_like(p); o.step(); assert p[0].item() < 0" > $BASE/ckpt/adam_cpu_build.log 2>&1 \
    || CPU_ADAM=failed
echo "DeepSpeed GPU Adam: $GPU_ADAM | CPU Adam: $CPU_ADAM (build logs: $BASE/ckpt/adam_*_build.log)"

mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
if [ $mem -ge 70000 ] && [ $GPU_ADAM = ok ]; then  # 80 GB: Adam on the GPU (RERUN_PAPER.md section 6)
  gpu="export ADAM_OFFLOAD= KEEP_VLLM=1 VLLM_MEM=0.25 NEED_RL=70000 EVAL_MEM=0.3"
elif [ $mem -ge 70000 ]; then                         # 80 GB, but GPU Adam did not build: CPU Adam (slower)
  gpu="export ADAM_OFFLOAD=1 KEEP_VLLM=1 VLLM_MEM=0.25 NEED_RL=40000 EVAL_MEM=0.3"
else                                                  # 24-48 GB
  gpu="export ADAM_OFFLOAD=1 KEEP_VLLM= VLLM_MEM=0.4 NEED_RL=18000 NEED_EVAL=12000 EVAL_MEM=0.5"
fi
cat > $BASE/treerl_env.sh <<EOF
# written by scripts/setup_cloud.sh on $(date): source it before every run on this instance
source $BASE/venv/bin/activate
export VENV=$BASE/venv HF_HOME=$BASE/hf UV_CACHE_DIR=$BASE/.uv-cache TORCH_EXTENSIONS_DIR=$BASE/torch_extensions
export CKPT=$BASE/ckpt USER=\${USER:-\$(id -un)}
$CUDA_ENV
$gpu
EOF
echo "== wrote $BASE/treerl_env.sh:"; cat $BASE/treerl_env.sh
[ $GPU_ADAM = ok ] || [ $CPU_ADAM = ok ] || { echo "SETUP FAILED: neither DeepSpeed Adam compiles (see the build logs)"; exit 1; }
echo "SETUP DONE. Next: source $BASE/treerl_env.sh && scripts/smoke_test.sh"
