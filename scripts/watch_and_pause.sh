#!/bin/bash
# Laptop side: wait until a cloud instance's pipeline (its tmux session, default "main") has ended, finished or
# failed, copy the results and logs back, then pause the instance with the JarvisLabs CLI (`jl`): GPU billing stops,
# the instance's disk is kept (and still billed). An unreachable instance is never paused: the check just repeats.
#   nohup caffeinate -is scripts/watch_and_pause.sh 551823 root@151.185.34.50 > ~/jarvis_watch.log 2>&1 &
# caffeinate keeps the Mac awake while it runs (a closed lid on battery still sleeps it; checks resume on wake).
ID=$1 HOST=$2 SESSION=${SESSION:-main} EVERY=${EVERY:-300}
cd "$(dirname "$0")/.." || exit 1
JL=$(command -v jl) || { echo "jl (JarvisLabs CLI) not found"; exit 1; }
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30"
OUT=mid_submission/results/rl_paper_ckpt_logs
while true; do
  state=$($SSH $HOST "tmux has-session -t $SESSION 2> /dev/null && echo running || echo ended" 2> /dev/null)
  [ "$state" = ended ] && break
  echo "$(date '+%F %T') $SESSION: ${state:-instance unreachable}"
  sleep $EVERY
done
echo "$(date '+%F %T') $SESSION has ended on the instance; copying results back"
mkdir -p $OUT
$SSH $HOST 'cat /home/next.log' > $OUT/next.log 2> /dev/null
$SSH $HOST 'cat /home/paper.log' > $OUT/paper.log 2> /dev/null
tail -5 $OUT/next.log; tail -12 $OUT/paper.log
rsync -az -e "$SSH" $HOST:/home/TreeRL/mid_submission/results/ mid_submission/results/
rsync -az -e "$SSH" --include '*/' --include '*.jsonl' --include 'run_config' --include '*.gpu_mem.log' \
    --include 'env_freeze.txt' --exclude '*' $HOST:/home/ckpt/ $OUT/
rsync -az -e "$SSH" $HOST:/home/TreeRL/mid_submission/data/train_30k_mixed_12k.jsonl mid_submission/data/
rsync -az -e "$SSH" $HOST:/home/TreeRL/mid_submission/data/train_30k_solve_rate_n8_12k.jsonl mid_submission/data/
echo "$(date '+%F %T') pausing instance $ID"
$JL pause $ID -y && echo "$(date '+%F %T') paused (resume with: jl resume $ID)"
