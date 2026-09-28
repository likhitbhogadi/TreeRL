#!/bin/bash
# Task 2 with TreeRL's own code on 500 unfiltered Omni-MATH problems (seed-0 sample).
#   nohup ./task2.sh > /tmp/likhit_task2.log 2>&1 &
# TreeRL's EPTree handles one problem at a time, so every config is split into K shards and run by
# SLOTS processes per GPU in parallel (each vLLM instance gets GPU_MEM of the card). Only GPUs that
# are free when the script starts are used. Safe to stop and re-run: shards resume by problem id.
set -x
cd "$(dirname "$0")"
source ~/likhit/.venv/bin/activate
export PYTHONUNBUFFERED=1
DATA=data/omni_math_500_seed0.jsonl
O=logs/task2
K=${K:-8}; SLOTS=${SLOTS:-4}; GPU_MEM=${GPU_MEM:-0.2}
mkdir -p $O
Q=$O/queue.txt

# name | runner args        (leaves = M + L*M*N*T; b0 = TreeRL's (M,0,0,0) i.i.d. chains)
CONFIGS="b0_64|--method b0 --M 64
b2_8-4-2-2|--method b2 --M 8 --N 4 --L 2 --T 2
b1_8-4-2-2|--method b1 --M 8 --N 4 --L 2 --T 2
b2_6-2-1-2|--method b2 --M 6 --N 2 --L 1 --T 2
b1_6-2-1-2|--method b1 --M 6 --N 2 --L 1 --T 2
b2_4-3-1-1|--method b2 --M 4 --N 3 --L 1 --T 1
b1_4-3-1-1|--method b1 --M 4 --N 3 --L 1 --T 1
b2_4-1-1-1|--method b2 --M 4 --N 1 --L 1 --T 1
b1_4-1-1-1|--method b1 --M 4 --N 1 --L 1 --T 1
b2_2-3-1-1|--method b2 --M 2 --N 3 --L 1 --T 1
b2_2-1-1-1|--method b2 --M 2 --N 1 --L 1 --T 1"

N_PROB=$(grep -c . $DATA)
fill_queue() {  # every (config, shard) whose shard file is not complete yet
  : > $Q
  echo "$CONFIGS" | while IFS='|' read name args; do
    [ -s $O/$name.jsonl ] && continue  # already merged
    for s in $(seq 0 $((K - 1))); do
      want=$(( (N_PROB - s + K - 1) / K ))
      have=$(cat $O/$name.jsonl.shard$s 2>/dev/null | grep -c .)
      [ "$have" -lt "$want" ] && echo "$name|$args|$s" >> $Q
    done
  done
}

fits() {  # enough free memory on GPU $1 for one more vLLM instance (+1 GiB margin)?
  nvidia-smi -i $1 --query-gpu=memory.free,memory.total --format=csv,noheader,nounits |
    awk -F', ' -v f=$GPU_MEM '{exit !($1 > f * $2 + 1024)}'
}

worker() {  # worker <gpu> <slot>
  while [ -s $Q ]; do
    until fits $1; do [ -s $Q ] || return; sleep 60; done  # the GPU is shared: wait, don't fail
    job=$(flock $Q.lock sh -c "head -n1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && return
    IFS='|' read name args s retry <<< "$job"
    echo "[gpu$1.$2] $(date +%T) start $name shard $s"
    if ! CUDA_VISIBLE_DEVICES=$1 python run_treerl.py $args --data $DATA --out $O/$name.jsonl \
        --shard $s --num_shards $K --gpu_mem $GPU_MEM >> $O/.$name.shard$s.log 2>&1; then
      if [ -z "$retry" ]; then  # e.g. another job grabbed the memory first: requeue once, back off
        echo "[gpu$1.$2] FAILED $name shard $s, requeued"
        flock $Q.lock sh -c "echo '$name|$args|$s|retry' >> $Q"
        sleep 300
      else
        echo "[gpu$1.$2] FAILED twice $name shard $s"
      fi
    fi
    echo "[gpu$1.$2] $(date +%T) done $name shard $s"
  done
}

GPUS=${GPUS:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 2000 {print $1}')}
[ -z "$GPUS" ] && { echo "no free GPU"; exit 1; }
for attempt in 1 2; do  # second pass retries shards that failed or were cut short
  fill_queue
  [ -s $Q ] || break
  for g in $GPUS; do
    for slot in $(seq 1 $SLOTS); do
      worker $g $slot &
      sleep 45  # stagger vLLM start-up so memory profiling doesn't collide
    done
  done
  wait
done

echo "$CONFIGS" | while IFS='|' read name args; do  # merge complete configs
  [ -s $O/$name.jsonl ] && continue
  n=$(cat $O/$name.jsonl.shard* 2>/dev/null | grep -c .)
  if [ "$n" -eq "$N_PROB" ]; then cat $O/$name.jsonl.shard* > $O/$name.jsonl && rm $O/$name.jsonl.shard*
  else echo "INCOMPLETE $name: $n/$N_PROB problems"; fi
done

python -m tree_alloc.task2_report --logs $O --out results/task2
CUDA_VISIBLE_DEVICES= python -m tree_alloc.task2_extra --logs $O --out results/task2
echo TASK2_DONE
