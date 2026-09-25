#!/bin/bash
# Task 2 (EPTree replication) on the lab box, one GPU, sequential.
#   nohup ./task2.sh > /tmp/likhit_task2.log 2>&1 &
# Leaves per problem = M + L*M*N*T. Budgets: 16 and 64 leaves (+ the paper's (6,2,1,2) = 30).
set -ex
cd "$(dirname "$0")"
source ~/likhit/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONUNBUFFERED=1
R="python -m tree_alloc.run"
DATA=../datasets/eval/olympiad_bench.jsonl
SELECT=logs/select_b0_olympiad.jsonl   # separate seed-100 B0 run, used only to pick problems

while pgrep -f "[o]ut $SELECT" > /dev/null; do sleep 30; done
SEL="--data $DATA --select_from $SELECT --acc_min 0.1 --acc_max 0.9 --limit 250"
O=logs/task2
mkdir -p $O

run() {  # run <b1|b2> M N L T
  $R gen --method $1 --M $2 --N $3 --L $4 --T $5 $SEL --out $O/$1_$2-$3-$4-$5.jsonl
}
run b2 4 3 1 1      # 16 leaves, wide
run b1 4 3 1 1
run b2 8 1 1 1      # 16 leaves, more chains
run b2 4 1 3 1      # 16 leaves, deep (3 iterations)
run b2 6 2 1 2      # 30 leaves, TreeRL's training config
run b1 6 2 1 2
$R gen --method b0 --n 64 $SEL --out $O/b0_64.jsonl   # i.i.d. pass@k curve, k = 1..64
run b2 16 3 1 1     # 64 leaves, wide
run b1 16 3 1 1
run b2 8 7 1 1      # 64 leaves, many forks per chain
run b2 8 1 7 1      # 64 leaves, deep (7 iterations)

python -m tree_alloc.task2_report --logs $O --out results/task2
echo TASK2_DONE
