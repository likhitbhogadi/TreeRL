#!/bin/bash
# Smoke test before the paper-setting runs, on pkgpu2 or the cloud instance (mid_submission/RERUN_PAPER.md, sec. 7).
# About 15 minutes. Every check prints PASS / WARN / FAIL; the script exits 1 if any check failed.
#   1. TreeRL: 3 steps x 2 prompts x 30 leaves with KL 1e-4: training, all-leaves path, KL reference model, tree
#      log, final save. Step 1 trains at lr 0 (warmup), so step 3 is the first with a changed policy (KL > 0).
#   2. ChainRL and GRPO: 1 step x 2 prompts x 16 chains each.
#   3. Evaluation: 4 problems of each benchmark, 2 samples each.
#   BENCH=1 then times 10 TreeRL steps at the paper batch (16 prompts x 30 leaves; ~25 min on an A100-80GB) and
#   compares the minutes per step with the plan in RERUN_PAPER.md section 6.
# GPU settings come from the environment (ADAM_OFFLOAD, KEEP_VLLM, VLLM_MEM, EVAL_MEM):
#   cloud:   source /home/treerl_env.sh && scripts/smoke_test.sh            (BENCH=1 for the timing run)
#   pkgpu2:  GPU=1 KEEP_VLLM=1 VLLM_MEM=0.25 scripts/smoke_test.sh
# ONLY="chainrl grpo" runs a subset of: treerl chainrl grpo eval.
cd "$(dirname "$0")/.." && ROOT=$(pwd)
source ${VENV:-$HOME/likhit/.venv}/bin/activate
GPU=${GPU:-0}
OUT=${OUT:-/tmp/$USER-treerl-smoke}
DATA=${DATA:-$ROOT/mid_submission/data/train_30k_mixed.jsonl}  # any training file: the smoke runs use < 200 prompts
KL=${KL:-1e-4}
ONLY=${ONLY:-treerl chainrl grpo eval}
rm -rf $OUT && mkdir -p $OUT
FAIL=0

echo "== environment"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
python - <<'EOF'
import importlib, platform
import torch
print(f"python {platform.python_version()} | torch {torch.__version__} (CUDA {torch.version.cuda}) | "
      f"GPU visible to torch: {torch.cuda.is_available()}")
print(" | ".join(f"{m} {importlib.import_module(m).__version__}" for m in ("vllm", "deepspeed", "transformers", "ray")))
EOF
echo "settings: GPU=$GPU ADAM_OFFLOAD=[${ADAM_OFFLOAD-1}] KEEP_VLLM=[${KEEP_VLLM:-}] VLLM_MEM=${VLLM_MEM:-0.35} KL=$KL"
echo "disk: $(df -h $OUT | awk 'NR==2 {print $4 " free at " $6}') | /dev/shm: $(df -h /dev/shm | awk 'NR==2 {print $2}')"

kill_run() {  # kill_run <run id> <launcher pid>: end a finished or stopped run completely. Ray's workers leave the
  # launcher's process group and can outlive the run holding GPU memory, so also kill every process of ours whose
  # environment carries the run's TREERL_RUN marker (other users' and other runs' processes are never touched)
  kill -9 -- -$2 2> /dev/null
  for p in $(grep -lzx "TREERL_RUN=$1" /proc/[0-9]*/environ 2> /dev/null | cut -d/ -f3); do kill -9 $p 2> /dev/null; done
}

train() {  # train <name> <method> <"M N L T"> <steps> <prompts> [VAR=value ...]
  local name=$1 method=$2 tree=$3 steps=$4 prompts=$5; shift 5
  echo "== $name: $method, $steps step(s) x $prompts prompts, tree ($tree)"
  local start=$(date +%s)
  local run_id=smoke-$name-$$
  setsid env TREERL_RUN=$run_id GPU=$GPU STEPS=$steps ROLLOUT=$prompts SAVE_STEPS=1000 KL=$KL TREE="$tree" TAG=qwen-smoke-$name \
      SAVE_DIR=$OUT/$name DATA=$DATA "$@" scripts/treerl-qwen1.5b-1gpu.sh > $OUT/$name.log 2>&1 &
  local pid=$!
  (while kill -0 $pid 2> /dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU; sleep 5; done \
     | sort -n | tail -1 > $OUT/$name.peak_mib) &
  wait $pid; local rc=$?
  kill_run $run_id $pid
  sleep 6; echo "   info  peak GPU memory during the run: $(cat $OUT/$name.peak_mib 2> /dev/null) MiB of $(nvidia-smi \
    --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU)"
  echo "   $(( $(date +%s) - start )) s, log: $OUT/$name.log"
  sleep 5  # nothing of ours may stay on the GPU between runs (leftover Ray workers once caused an out-of-memory)
  local left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU | while read p; do
    [ "$(ps -o user= -p $p 2> /dev/null)" = "$USER" ] && echo $p; done | wc -l)
  [ $left -eq 0 ] && echo "   PASS  no processes of ours left on the GPU" || { echo "   FAIL  $left processes of ours left on the GPU"; FAIL=1; }
  python scripts/smoke_check.py train $method $OUT/$name $OUT/$name.log $rc $steps $prompts "$tree" $KL || FAIL=1
  rm -f $OUT/$name/*.safetensors  # the saved model (3 GB): only its existence was checked
  sleep 20  # let the GPU memory of the finished run go before the next one starts
}

has() { [[ " $ONLY " == *" $1 "* ]]; }
has treerl && train treerl treerl "6 2 1 2" 3 2
has chainrl && train chainrl chainrl "16 0 0 0" 1 2
has grpo && train grpo grpo "16 0 0 0" 1 2 ADV=grpo

if has eval; then
echo "== evaluation: 4 problems per benchmark, 2 samples each"
(cd mid_submission && CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 python solve_rate.py \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct --sample 4 --n 2 --temperature 1.0 --gpu_mem ${EVAL_MEM:-0.28} \
    --data ../datasets/eval/MATH500.jsonl ../datasets/eval/aimo-validation-amc.jsonl data/omni_math_500_seed0.jsonl \
           ../datasets/eval/olympiad_bench.jsonl \
    --summary $OUT/eval.csv --out $OUT/eval.jsonl > $OUT/eval.log 2>&1)
rc=$?
echo "   log: $OUT/eval.log"
python scripts/smoke_check.py eval $OUT/eval.csv $OUT/eval.jsonl $rc 4 4 || FAIL=1
fi

if [ -n "$BENCH" ]; then
  sleep 20
  train bench treerl "6 2 1 2" 10 16
fi

if [ $FAIL -eq 0 ]; then echo "SMOKE TEST PASSED"; else echo "SMOKE TEST FAILED: see the FAIL lines and logs above"; exit 1; fi
