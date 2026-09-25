#!/bin/bash
# Task 2 replication set: 500 unfiltered Omni-MATH problems (seed-0 sample, like the paper's
# Omni-MATH-500). One worker per GPU pulls jobs from a shared queue; a worker only starts a job
# when its GPU is free, so a GPU busy with someone else's job is skipped until it frees up.
#   nohup ./task2_omni.sh > /tmp/likhit_task2_omni.log 2>&1 &
# Safe to stop and re-run at any time: every job resumes by problem id (saved every 256 problems),
# and finished jobs exit without loading the model.
set -x
cd "$(dirname "$0")"
source ~/likhit/.venv/bin/activate
export PYTHONUNBUFFERED=1
DATA=data/omni_math_500_seed0.jsonl
O=logs/task2_omni
mkdir -p $O
Q=$O/queue.txt

# largest first; leaves = M + L*M*N*T
cat > $Q <<EOF
gen --method b0 --n 64 --out $O/b0_64.jsonl
gen --method b2 --M 8 --N 4 --L 2 --T 2 --out $O/b2_8-4-2-2.jsonl
gen --method b1 --M 8 --N 4 --L 2 --T 2 --out $O/b1_8-4-2-2.jsonl
gen --method b2 --M 6 --N 2 --L 1 --T 2 --out $O/b2_6-2-1-2.jsonl
gen --method b1 --M 6 --N 2 --L 1 --T 2 --out $O/b1_6-2-1-2.jsonl
gen --method b2 --M 4 --N 3 --L 1 --T 1 --out $O/b2_4-3-1-1.jsonl
gen --method b1 --M 4 --N 3 --L 1 --T 1 --out $O/b1_4-3-1-1.jsonl
gen --method b2 --M 4 --N 1 --L 1 --T 1 --out $O/b2_4-1-1-1.jsonl
gen --method b1 --M 4 --N 1 --L 1 --T 1 --out $O/b1_4-1-1-1.jsonl
gen --method b2 --M 2 --N 3 --L 1 --T 1 --out $O/b2_2-3-1-1.jsonl
gen --method b2 --M 2 --N 1 --L 1 --T 1 --out $O/b2_2-1-1-1.jsonl
EOF

# A GPU counts as free only if it stays empty for 5 consecutive checks a minute apart: other users
# run jobs back to back, and grabbing a gap between them makes one side OOM (this happened once).
gpu_free() {
  for _ in 1 2 3 4 5; do
    [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] || return 1
    sleep 60
  done
}

worker() {
  local gpu=$1
  while [ -s $Q ]; do
    until gpu_free $gpu; do
      [ -s $Q ] || return 0  # queue drained while waiting for a busy GPU
      sleep 60
    done
    job=$(flock $Q.lock sh -c "head -n1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    echo "[worker gpu$gpu] $(date +%T) start: $job"
    if ! CUDA_VISIBLE_DEVICES=$gpu python -m tree_alloc.run ${job%%#*} --data $DATA > $O/gpu$gpu.$(date +%s).log 2>&1; then
      if [[ "$job" == *"#retry"* ]]; then
        echo "[worker gpu$gpu] FAILED twice, giving up: $job"
      else  # one retry; runs resume by problem id, so re-running is safe
        echo "[worker gpu$gpu] FAILED, requeued once: $job"
        flock $Q.lock sh -c "echo '$job #retry' >> $Q"
        sleep 300
      fi
    fi
    echo "[worker gpu$gpu] $(date +%T) done: $job"
  done
}

worker 1 &
sleep 90  # let GPU 1 claim the first job before GPU 0 is considered
worker 0 &
wait

python -m tree_alloc.task2_report --logs $O --out results/task2_omni
echo TASK2_OMNI_DONE
CUDA_VISIBLE_DEVICES= python -m tree_alloc.task2_extra --logs $O --out results/task2_omni
echo EXTRA_OMNI_DONE
