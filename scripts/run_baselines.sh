#!/bin/bash
# Unattended baselines on the shared GPU box, at the reduced budget of mid_submission/RL_CHANGES.md:
#   1. base-model eval   2-4. TreeRL (6,2,1,2), ChainRL (8 chains), GRPO (8 chains), each + eval of its checkpoints
#   5. 8-sample eval of steps 20/40   6-8. longer TreeRL, ChainRL, GRPO runs (150 steps, lr 5e-6) + their eval
# Each stage waits for a GPU with enough free memory. Finished stages are skipped, so re-running continues
# where it stopped. An RL run that dies (e.g. OOM because another user's job grew) or makes no progress for
# 45 min waits for memory again and resumes from its newest checkpoint (3 tries).
#   nohup scripts/run_baselines.sh > /tmp/likhit_baselines.log 2>&1 &
cd "$(dirname "$0")/.." && ROOT=$(pwd)
source ~/likhit/.venv/bin/activate
NEED_RL=${NEED_RL:-21000}      # MiB free to start RL: measured peak 18.4 GB (1.5B, VLLM_MEM=0.3) + margin
NEED_EVAL=${NEED_EVAL:-14000}  # vLLM at 0.28 x 46 GB
COMMON="ROLLOUT=8 NUM_TRACE=8 MAX_LEN=2048 STEPS=${STEPS:-40} SAVE_STEPS=10 VLLM_MEM=0.3"
DATA=$ROOT/mid_submission/data/train_30k_mixed.jsonl
EVAL_SETS="../datasets/eval/MATH500.jsonl ../datasets/eval/aimo-validation-amc.jsonl data/omni_math_500_seed0.jsonl"
DONE=$ROOT/ckpt/.done
mkdir -p $DONE

free_gpu() {  # block until some GPU has >= $1 MiB free; print its index
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v n=$1 '$2 >= n {print $1; exit}')
    [ -n "$g" ] && { echo $g; return; }
    sleep 120
  done
}

our_gpu_mem() {  # MiB used by this user's processes on GPU $1
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i $1 |
    while IFS=', ' read p m; do [ "$(ps -o user= -p $p)" = "$USER" ] && echo $m; done | awk '{s += $1} END {print s + 0}'
}

evaluate() {  # evaluate <model path or id> <name> [solve_rate.py args]: greedy by default -> mid_submission/results/rl_eval.csv
  [ -f $DONE/eval_$2 ] && return
  for try in 1 2 3; do
    g=$(free_gpu $NEED_EVAL)
    echo "$(date +%T) eval $2 on GPU $g"
    if (cd mid_submission && CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 python solve_rate.py --model $1 --temperature 0 \
          --gpu_mem 0.28 --summary $DONE/eval_$2.csv --data $EVAL_SETS "${@:3}" > $DONE/eval_$2.log 2>&1); then
      tail -n +$([ -f mid_submission/results/rl_eval.csv ] && echo 2 || echo 1) $DONE/eval_$2.csv >> mid_submission/results/rl_eval.csv
      touch $DONE/eval_$2
      sleep 60  # let vLLM free its memory: a new engine's startup profiling fails if free memory changes
      return
    fi
    echo "$(date +%T) eval $2 failed (see $DONE/eval_$2.log)"; rm -f $DONE/eval_$2.csv; sleep 300
  done
}

train() {  # train <tag> <"M N L T"> [VAR=value overrides, e.g. STEPS=150 LR=5e-6]
  local dir=$ROOT/ckpt/$1
  for try in 1 2 3; do
    [ -f $dir/model.safetensors ] && return  # the final model is saved at the end of a completed run
    g=$(free_gpu $NEED_RL)
    resume=$(ls -d $dir/_actor_global_step* 2>/dev/null | head -1)
    echo "$(date +%T) train $1 (try $try) on GPU $g${resume:+, resuming}"
    env $COMMON GPU=$g TREE="$2" TAG=$1 SAVE_DIR=$dir DATA=$DATA RESUME=${resume:+1} "${@:3}" \
      scripts/treerl-qwen1.5b-1gpu.sh > $ROOT/ckpt/$1.try$try.log 2>&1 &
    pid=$! start=$(date +%s)
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      echo "$(date +%T) $(our_gpu_mem $g)" >> $ROOT/ckpt/$1.gpu_mem.log  # measured VRAM (MiB), for RL_CHANGES.md
      last=$(stat -c %Y $dir/train_log.jsonl 2>/dev/null || echo 0)
      [ $last -lt $start ] && last=$start
      if [ $(( $(date +%s) - last )) -gt 2700 ]; then
        echo "$(date +%T) $1: no training step for 45 min, restarting"; pkill -P $pid; kill $pid
      fi
    done
    ray stop --force > /dev/null 2>&1
    sleep 60
  done
  echo "$(date +%T) train $1: not finished after 3 tries"
}

[ -s $DATA ] || { echo "missing $DATA (run mid_submission/solve_rate.py --keep_mixed first)"; exit 1; }
evaluate Qwen/Qwen2.5-Math-1.5B-Instruct base
# tags contain "qwen": the released code picks the model family from the checkpoint path
# tag | TREE | extra overrides. GRPO = ChainRL's 8 i.i.d. chains + group-normalized advantage (GRPO_CHANGES.md)
for run in "qwen1.5b-treerl-6-2-1-2|6 2 1 2|" "qwen1.5b-chainrl-8|8 0 0 0|" "qwen1.5b-grpo-8|8 0 0 0|ADV=grpo"; do
  IFS='|' read tag tree extra <<< "$run"
  train $tag "$tree" $extra
  for c in $(ls -d $ROOT/ckpt/$tag/_actor_global_step* 2>/dev/null | sort -V); do  # right after training
    evaluate $c ${tag}_$(basename $c)
  done
done

# Lower-noise eval: 8 samples per problem at the training temperature (greedy scores moved within noise)
SAMPLED="--n 8 --temperature 1.0"
evaluate Qwen/Qwen2.5-Math-1.5B-Instruct base_n8 $SAMPLED
for tag in qwen1.5b-treerl-6-2-1-2 qwen1.5b-chainrl-8 qwen1.5b-grpo-8; do
  for step in 20 40; do evaluate $ROOT/ckpt/$tag/_actor_global_step$step ${tag}_step${step}_n8 $SAMPLED; done
done

# Longer runs with a larger step, identical settings for all three methods (40 steps at lr 1.5e-6 did not
# move greedy accuracy); each run is scored right after it finishes
for run in "qwen1.5b-treerl-6-2-1-2-lr5e-6-150|6 2 1 2|" "qwen1.5b-chainrl-8-lr5e-6-150|8 0 0 0|" \
           "qwen1.5b-grpo-8-lr5e-6-150|8 0 0 0|ADV=grpo"; do
  IFS='|' read tag tree extra <<< "$run"
  train $tag "$tree" STEPS=150 LR=5e-6 SAVE_STEPS=50 $extra
  for c in $(ls -d $ROOT/ckpt/$tag/_actor_global_step* 2>/dev/null | sort -V); do
    evaluate $c ${tag}_$(basename $c)_n8 $SAMPLED
  done
done
echo "$(date +%T) BASELINES_DONE"
