#!/bin/bash
# TreeRL (EPTree + process supervision, REINFORCE) on ONE GPU: Qwen2.5-Math-1.5B-Instruct.
# Same algorithm flags as scripts/treerl-qw14b.sh; actor (DeepSpeed ZeRO-2, CPU Adam), and the vLLM engine
# share the GPU. See mid_submission/RL_CHANGES.md.
#   GPU=0 STEPS=100 nohup scripts/treerl-qwen1.5b-1gpu.sh > /tmp/treerl_rl.log 2>&1 &
# Paper settings (Sec. 4.1) are the defaults; only the tree shape differs between methods (TREE="M N L T"):
#   TreeRL  (EPTree 6,2,1,2, all 30 leaves trained on, batch 16 x 30 = 480):  TREE="6 2 1 2"
#   ChainRL (16 i.i.d. chains, same generated tokens, batch 16 x 16 = 256):   TREE="16 0 0 0" TAG=...-chainrl
#   GRPO    (same 16 chains, group-normalized advantage):                       TREE="16 0 0 0" ADV=grpo TAG=...-grpo
# The full paper-setting pipeline is scripts/run_paper.sh (mid_submission/RERUN_PAPER.md).
# Earlier reduced budget (RL_CHANGES.md): ROLLOUT=8 NUM_TRACE=8 TREE="6 2 1 2" (or "8 0 0 0") MAX_LEN=2048 STEPS=40
set -x
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd $ROOT
source ${VENV:-$HOME/likhit/.venv}/bin/activate  # VENV: the Python environment (e.g. on a cloud instance)

GPU=${GPU:-0}
STEPS=${STEPS:-100}          # training steps; each step = ROLLOUT prompts x NUM_TRACE leaves
ROLLOUT=${ROLLOUT:-16}       # prompts per step (paper: 16)
TREE=(${TREE:-6 2 1 2})      # EPTree M N L T; "16 0 0 0" = ChainRL (M i.i.d. chains)
LEAVES=$(( TREE[0] * (1 + TREE[1] * TREE[2] * TREE[3]) ))  # leaves per tree: M (1 + N L T); (6,2,1,2) -> 30
NUM_TRACE=${NUM_TRACE:-$LEAVES}  # leaves per tree used for training (paper: all 30; its Table 3: 16 of 30 costs 3.2 pts)
[ $NUM_TRACE -le $LEAVES ] || { echo "NUM_TRACE=$NUM_TRACE > $LEAVES leaves of a (${TREE[*]}) tree"; exit 1; }
# All leaves: train on every leaf in tree order. A subset (NUM_TRACE < LEAVES) is drawn at random but always
# includes one correct leaf, which biases the mean advantage upward (~+0.06 per response at 8 of 30).
ALL_LEAVES=$([ $NUM_TRACE -eq $LEAVES ] && echo --use_all_terminals)
KL=${KL:-1e-4}               # KL coefficient: paper text 1e-4 (released script 0); > 0 loads a reference model
SEED=${SEED:-42}             # vLLM sampling, data order and torch seeds (repetitions: SEED=1, 2, ...)
ADAM_OFFLOAD=${ADAM_OFFLOAD-1}  # empty: Adam states on the GPU (~28 GB for 1.5B; fits on 80 GB). Default: CPU Adam,
                             # which also moves the gradient to the CPU after every response (slow, but fits 24-48 GB)
INFER_BS=${INFER_BS:-2}      # batch of the logprob pass; must divide NUM_TRACE (the remainder of a tree is dropped)
[ $((NUM_TRACE % INFER_BS)) -eq 0 ] || { echo "INFER_BS=$INFER_BS must divide NUM_TRACE=$NUM_TRACE"; exit 1; }
KEEP_VLLM=${KEEP_VLLM:-}        # non-empty: vLLM keeps its GPU memory through training (no sleep; for shared GPUs)
ADV=${ADV:-treerl}            # advantage: treerl (tree values) | grpo (needs TREE="G 0 0 0"; mid_submission/GRPO_CHANGES.md)
MAX_LEN=${MAX_LEN:-3072}     # max response tokens (Task 2: median 682, 1.1% over 2048)
RESUME=${RESUME:-}             # non-empty: resume from the newest $SAVE_DIR/_actor_global_step* (must exist)
DATA=$(realpath "${DATA:-$ROOT/datasets/train/train_30k.jsonl}")  # absolute: Ray workers have their own cwd
TAG=${TAG:-qwen1.5b-math-treerl-6-2-1-2}
SAVE_DIR=${SAVE_DIR:-$ROOT/ckpt/$TAG}
mkdir -p $SAVE_DIR
# the trainer takes min(max_samples, rows) prompts and stops when they run out, so a short file ends the run early
[ $(wc -l < $DATA) -ge $((STEPS * ROLLOUT)) ] || { echo "$DATA has fewer than STEPS x ROLLOUT = $((STEPS * ROLLOUT)) rows"; exit 1; }

export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 RAY_DEDUP_LOGS=0
# Ray starts implicitly; "local" makes it start this run's own cluster rather than attach to another run's
# (two runs side by side on GPU0 and GPU1 must not share one)
export RAY_ADDRESS=local

python train_reinforce_ray.py \
    --actor_num_nodes 1 --actor_num_gpus_per_node 1 \
    --ref_num_nodes 1 --ref_num_gpus_per_node 1 \
    --reward_num_nodes 0 \
    --vllm_num_engines 1 --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization ${VLLM_MEM:-0.35} --enable_prefix_caching ${KEEP_VLLM:+--vllm_keep_memory} \
    --pretrain ${MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct} \
    --reward_pretrain ${MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct} \
    --save_path $SAVE_DIR --ckpt_path $SAVE_DIR --save_steps ${SAVE_STEPS:-20} \
    --micro_train_batch_size 1 \
    --train_batch_size $((ROLLOUT * NUM_TRACE)) \
    --micro_rollout_batch_size $ROLLOUT \
    --rollout_batch_size $ROLLOUT \
    --inference_batch_size $INFER_BS \
    --num_episodes 1 \
    --max_samples $((STEPS * ROLLOUT)) \
    --prompt_max_len 1024 \
    --generate_max_len $MAX_LEN \
    --zero_stage 2 ${ADAM_OFFLOAD:+--adam_offload} --bf16 --gradient_checkpointing \
    --actor_learning_rate ${LR:-1.5e-6} --lr_scheduler_type cosine --min_actor_learning_rate_lr 1 --l2 0.1 \
    --init_kl_coef $KL --seed $SEED \
    --prompt_data $DATA,1 \
    --input_key text --label_key label --source_key data_type \
    --top_p 0.95 --temperature 1.0 \
    --num_trace_per_sample $NUM_TRACE \
    --task_type qwen-math-reinforce \
    --remote_rm_url $ROOT/scripts/remote_reward_url.json \
    --advantage_estimator $ADV \
    --use_mcts --use_entropy_tree --m ${TREE[0]} --n ${TREE[1]} --l ${TREE[2]} --t ${TREE[3]} \
    --process_supervision --use_state_value_reward --use_pure_binary \
    --use_weighted_value --weighted_value_style sqrt $ALL_LEAVES \
    --mask_repeated_samples \
    --correct_bonus_ratio 1 --correct_bonus_threshold 0 \
    --perf ${RESUME:+--resume} \
    --wandb_run_name $TAG
