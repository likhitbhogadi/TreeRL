#!/bin/bash
# Task 1 smoke test on the lab box: 20 MATH500 problems, one GPU.
#   nohup ./smoke.sh > /tmp/likhit_smoke.log 2>&1 &
set -ex
cd "$(dirname "$0")"
source ~/likhit/.venv/bin/activate          # puts ninja etc. on PATH
export CUDA_VISIBLE_DEVICES=${GPU:-1}
D=../datasets/eval/MATH500.jsonl
mkdir -p logs
rm -f logs/smoke_*.jsonl
python run_treerl.py --method b0 --M 4 --data $D --limit 20 --out logs/smoke_b0.jsonl
python run_treerl.py --method b2 --M 2 --N 1 --L 1 --T 1 --data $D --limit 20 --out logs/smoke_b2.jsonl
python -m tree_alloc.run logs/smoke_b0.jsonl logs/smoke_b2.jsonl
python -m tree_alloc.checks --log logs/smoke_b0.jsonl
echo SMOKE_DONE
