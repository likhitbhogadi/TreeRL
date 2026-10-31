#!/bin/bash
# Paper-setting reruns of TreeRL, ChainRL and GRPO (TreeRL paper Sec. 4.1; mid_submission/RERUN_PAPER.md).
# Every step: 16 prompts (ROLLOUT), one optimizer step, lr 1.5e-6 (10-step warmup, then constant), KL 1e-4,
# responses up to 3072 tokens. Run names end in -seed$SEED; each run folder keeps its settings in run_config, and a
# folder whose settings differ is never resumed or skipped as done.
#   treerl   EPTree (6,2,1,2): 30 leaves per prompt, all trained on (batch 480)
#   chainrl  16 i.i.d. chains per prompt, about the tree's generated tokens (batch 256)
#   grpo     the same 16 chains, GRPO-style group-normalized advantage (our baseline, not in the paper; batch 256)
# After each run, every EVAL_EVERY-th checkpoint is scored greedy and with 8 samples per problem on MATH500, AMC,
# Omni-MATH-500 and OlympiadBench (per-problem files, so tree_alloc/rl_report.py can compute paired CIs vs the base).
# Results go to their own files (results/rl_paper_*), apart from the earlier runs'.
# Stages wait for GPU memory; finished stages are skipped, so re-running continues where it stopped; a run that dies
# or stalls resumes from its newest checkpoint (weights only: fresh Adam; logged as resumed_from and reported).
# Ends with PAPER_RUNS_DONE, or PAPER_RUNS_INCOMPLETE plus the unfinished stages and exit status 1.
#   nohup scripts/run_paper.sh > /tmp/likhit_paper.log 2>&1 &                                # all three in turn
#   ONLY=treerl GPU=0 nohup scripts/run_paper.sh > /tmp/likhit_paper_tree.log 2>&1 &          # or one GPU each
#   ONLY="chainrl grpo" GPU=1 nohup scripts/run_paper.sh > /tmp/likhit_paper_chain.log 2>&1 &
# GPU settings (mid_submission/RERUN_PAPER.md, section 5):
#   80 GB:     ADAM_OFFLOAD= KEEP_VLLM=1 VLLM_MEM=0.3 NEED_RL=70000 EVAL_MEM=0.3
#   24 GB:     ADAM_OFFLOAD=1 KEEP_VLLM= VLLM_MEM=0.4 NEED_RL=18000 NEED_EVAL=12000 EVAL_MEM=0.5
#   pkgpu2:    defaults (46 GB, shared with other users)
cd "$(dirname "$0")/.." && ROOT=$(pwd)
source ${VENV:-$HOME/likhit/.venv}/bin/activate  # VENV: the Python environment (e.g. on a cloud instance)
STEPS=${STEPS:-200}            # paper curves run ~300 steps; TreeRL pulls ahead from ~100
ROLLOUT=${ROLLOUT:-16}         # prompts per step (paper 16; 8 halves the cost, same per-prompt settings)
LR=${LR:-1.5e-6}
KL=${KL:-1e-4}                 # paper's KL coefficient; loads a reference model (the released script used 0)
SEED=${SEED:-42}               # repetitions: SEED=1, 2, ... (part of the run names)
ONLY=${ONLY:-treerl chainrl grpo}
# MODEL: the starting model. Each model has its own run names, training data, stage markers and result files;
# the Instruct model keeps the original names (qwen1.5b-paper-*, results/rl_paper_*).
MODEL=${MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}
case $MODEL in
  Qwen/Qwen2.5-Math-1.5B-Instruct) MNAME=qwen1.5b SUF= ;;
  Qwen/Qwen2.5-Math-1.5B)          MNAME=qwen1.5b-mathbase SUF=_mathbase ;;  # no RL by Qwen (the paper starts from SFT)
  *) [ -n "$MNAME" ] && [ -n "$SUF" ] || { echo "MODEL=$MODEL: also set MNAME (run-name prefix, must contain qwen) and SUF"; exit 1; } ;;
esac
GPU=${GPU:-}                   # pin to one GPU (to run two instances side by side); empty: any GPU with free memory
DATA=${DATA:-$ROOT/mid_submission/data/train_30k_mixed_12k$SUF.jsonl}  # RERUN_PAPER.md, step 1 (built with $MODEL)
CKPT=${CKPT:-$([ -d /scratch ] && echo /scratch/$USER/treerl_ckpt || echo $ROOT/ckpt)}  # ~3 GB per checkpoint
SAVE_STEPS=${SAVE_STEPS:-25}   # bounds the work lost to a crash (~2 h of TreeRL)
EVAL_EVERY=${EVAL_EVERY:-50}
NEED_RL=${NEED_RL:-35000}      # MiB free to start training (earlier long runs peaked at 21-31 GB with KEEP_VLLM=1,
                               # + 3 GB for the KL reference model)
NEED_EVAL=${NEED_EVAL:-14000}
EVAL_MEM=${EVAL_MEM:-0.28}     # vLLM's share of the GPU during evaluation
TRIES=${TRIES:-6}
STALL=${STALL:-3600}           # seconds without a training step before a run is restarted (a TreeRL step is ~5 min)
# ADAM_OFFLOAD is read by the launcher from the environment (default: CPU Adam)
COMMON="MODEL=$MODEL ROLLOUT=$ROLLOUT STEPS=$STEPS MAX_LEN=3072 LR=$LR KL=$KL SEED=$SEED SAVE_STEPS=$SAVE_STEPS
        KEEP_VLLM=${KEEP_VLLM-1} VLLM_MEM=${VLLM_MEM:-0.25}"
EVAL_SETS="../datasets/eval/MATH500.jsonl ../datasets/eval/aimo-validation-amc.jsonl data/omni_math_500_seed0.jsonl
           ../datasets/eval/olympiad_bench.jsonl"
DONE=$CKPT/.done_paper$SUF  # stage markers live with the checkpoints they describe
PP=results/rl_paper${SUF}_per_problem          # 8-sample per-problem results (relative to mid_submission/), paired vs base_n8
PPG=results/rl_paper${SUF}_per_problem_greedy  # greedy per-problem results, paired vs base_greedy
SUMMARY=results/rl_paper${SUF}_eval.csv        # one row per (model, benchmark), as results/rl_eval.csv for the earlier runs
mkdir -p $DONE $CKPT mid_submission/$PP mid_submission/$PPG
FAILED=()  # stages that did not finish, reported at the end
TAGS=()

rows=$(wc -l < $DATA 2>/dev/null || echo 0)
[ $rows -ge $((STEPS * ROLLOUT)) ] || { echo "$DATA: $rows rows, need STEPS x ROLLOUT = $((STEPS * ROLLOUT)) (RERUN_PAPER.md, step 1)"; exit 1; }

gpu_with() {  # gpu_with <MiB>: block until $GPU (or any GPU) has that much free memory; print its index
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
        awk -F', ' -v n=$1 -v pin="$GPU" '(pin == "" || $1 == pin) && $2 >= n {print $1; exit}')
    [ -n "$g" ] && { echo $g; return; }
    sleep 120
  done
}

our_gpu_mem() {  # MiB used by this user's processes on GPU $1
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i $1 |
    while IFS=', ' read p m; do [ "$(ps -o user= -p $p)" = "$USER" ] && echo $m; done | awk '{s += $1} END {print s + 0}'
}

evaluate() {  # evaluate <model path or id> <name> <per-problem dir> [solve_rate.py args] -> mid_submission/$SUMMARY
  [ -f $DONE/eval_$2 ] && return
  # two instances (GPU=0, GPU=1) both start with the base-model evaluations: one runs each, the other waits for
  # its result. A lock whose owner has died is taken over.
  until mkdir $DONE/lock_$2 2> /dev/null; do
    [ -f $DONE/eval_$2 ] && return
    kill -0 $(cat $DONE/lock_$2/pid 2> /dev/null || echo 0) 2> /dev/null || rm -rf $DONE/lock_$2
    sleep 60
  done
  echo $$ > $DONE/lock_$2/pid
  [ -f $DONE/eval_$2 ] && { rm -rf $DONE/lock_$2; return; }  # the other instance finished it while we waited
  local status=1
  for try in 1 2 3; do
    g=$(gpu_with $NEED_EVAL)
    echo "$(date +%T) eval $2 on GPU $g"
    rm -f mid_submission/$3/$2.jsonl
    if (cd mid_submission && CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 python solve_rate.py --model $1 --temperature 0 \
          --gpu_mem $EVAL_MEM --summary $DONE/eval_$2.csv --out $3/$2.jsonl --data $EVAL_SETS "${@:4}" > $DONE/eval_$2.log 2>&1); then
      tail -n +$([ -f mid_submission/$SUMMARY ] && echo 2 || echo 1) $DONE/eval_$2.csv >> mid_submission/$SUMMARY
      touch $DONE/eval_$2
      status=0
      break
    fi
    echo "$(date +%T) eval $2 failed (see $DONE/eval_$2.log)"; rm -f $DONE/eval_$2.csv; sleep 300
  done
  rm -rf $DONE/lock_$2
  sleep 60  # let vLLM free its memory: a new engine's startup profiling fails if free memory changes
  return $status
}

kill_run() {  # kill_run <run id> <launcher pid>: end a finished or stopped run completely. Ray's workers leave the
  # launcher's process group and can outlive the run holding GPU memory, so also kill every process of ours whose
  # environment carries the run's TREERL_RUN marker (other users' and other runs' processes are never touched)
  kill -9 -- -$2 2> /dev/null
  for p in $(grep -lzx "TREERL_RUN=$1" /proc/[0-9]*/environ 2> /dev/null | cut -d/ -f3); do kill -9 $p 2> /dev/null; done
}

train() {  # train <tag> <"M N L T"> [VAR=value overrides]; 1: not finished after $TRIES tries, 2: settings clash
  local dir=$CKPT/$1
  # the settings that define the run: a folder holding a run with other settings is never resumed or reused
  local cfg="tree=$2 ${*:3} steps=$STEPS rollout=$ROLLOUT lr=$LR kl=$KL seed=$SEED max_len=3072 \
data=$(basename $DATA) data_cksum=$(cksum < $DATA | cut -d' ' -f1)${SUF:+ model=$MODEL}"
  mkdir -p $dir
  if [ -f $dir/run_config ]; then
    if [ "$(cat $dir/run_config)" != "$cfg" ]; then
      echo "$(date +%T) train $1: $dir holds a run with other settings (use another CKPT, or SEED)"
      echo "  folder: $(cat $dir/run_config)"; echo "  wanted: $cfg"
      return 2
    fi
  elif [ -f $dir/model.safetensors ] || ls -d $dir/_actor_global_step* > /dev/null 2>&1; then
    echo "$(date +%T) train $1: $dir holds checkpoints of unknown settings (no run_config); not reusing them"
    return 2
  else
    echo "$cfg" > $dir/run_config
  fi
  for try in $(seq 1 $TRIES); do
    [ -f $dir/model.safetensors ] && return  # the final model is saved at the end of a completed run
    g=$(gpu_with $NEED_RL)
    resume=$(ls -d $dir/_actor_global_step* 2>/dev/null | head -1)
    echo "$(date +%T) train $1 (try $try) on GPU $g${resume:+, resuming}"
    # own session: the run (driver, Ray's gcs/raylet/workers) can be killed as one process group
    local run_id=$1-try$try-$$
    setsid env TREERL_RUN=$run_id $COMMON GPU=$g TREE="$2" TAG=$1 SAVE_DIR=$dir DATA=$DATA RESUME=${resume:+1} "${@:3}" \
      scripts/treerl-qwen1.5b-1gpu.sh > $CKPT/$1.try$try.log 2>&1 &
    pid=$! start=$(date +%s)
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      echo "$(date +%T) $(our_gpu_mem $g)" >> $CKPT/$1.gpu_mem.log
      last=$(stat -c %Y $dir/train_log.jsonl 2>/dev/null || echo 0)
      [ $last -lt $start ] && last=$start
      if [ $(( $(date +%s) - last )) -gt $STALL ]; then
        echo "$(date +%T) $1: no training step for $STALL s, restarting"; kill -- -$pid
      fi
    done
    kill_run $run_id $pid
    # unpinned = the only instance: also stop any Ray process that left the group (as run_baselines.sh did);
    # pinned instances skip this, since `ray stop` would kill the other instance's run too
    [ -z "$GPU" ] && ray stop --force > /dev/null 2>&1
    sleep 60
  done
  [ -f $dir/model.safetensors ] || { echo "$(date +%T) train $1: not finished after $TRIES tries"; return 1; }
}

SAMPLED="--n 8 --temperature 1.0"
evaluate $MODEL base_greedy $PPG || FAILED+=("eval base_greedy")
evaluate $MODEL base_n8 $PP $SAMPLED || FAILED+=("eval base_n8")
# tags contain "qwen": the released code picks the model family from the checkpoint path
for method in $ONLY; do
  case $method in
    treerl)  tag=$MNAME-paper-treerl-6-2-1-2-seed$SEED; tree="6 2 1 2"; extra= ;;
    chainrl) tag=$MNAME-paper-chainrl-16-seed$SEED;     tree="16 0 0 0"; extra= ;;
    grpo)    tag=$MNAME-paper-grpo-16-seed$SEED;        tree="16 0 0 0"; extra=ADV=grpo ;;
    *) echo "unknown method $method"; FAILED+=("unknown method $method"); continue ;;
  esac
  TAGS+=($tag)
  train $tag "$tree" $extra; rc=$?
  [ $rc -eq 1 ] && FAILED+=("train $tag")
  [ $rc -eq 2 ] && { FAILED+=("train $tag: folder holds other settings"); continue; }  # don't score that run
  # score the checkpoints that exist even if training stopped early (it is reported above either way)
  for step in $(seq $EVAL_EVERY $EVAL_EVERY $STEPS); do
    c=$CKPT/$tag/_actor_global_step$step
    if [ ! -d $c ]; then
      [ -f $CKPT/$tag/model.safetensors ] && FAILED+=("missing checkpoint $tag step $step")
      continue
    fi
    evaluate $c ${tag}_$(basename $c)_greedy $PPG || FAILED+=("eval ${tag} step $step greedy")
    evaluate $c ${tag}_$(basename $c)_n8 $PP $SAMPLED || FAILED+=("eval ${tag} step $step n8")
  done
done
for tag in "${TAGS[@]}"; do  # resumed runs continued with a fresh optimizer state: disclose them
  r=$(grep -o '"resumed_from": [1-9][0-9]*' $CKPT/$tag/train_log.jsonl 2> /dev/null | awk '{print $2}' | sort -un | tr '\n' ' ')
  [ -n "$r" ] && echo "$(date +%T) note: $tag resumed from step(s) $r(fresh optimizer state)"
done
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "$(date +%T) PAPER_RUNS_INCOMPLETE ($ONLY):"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "$(date +%T) PAPER_RUNS_DONE ($ONLY)"
