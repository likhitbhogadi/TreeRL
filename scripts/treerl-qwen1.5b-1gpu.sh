#!/bin/bash
# TreeRL (EPTree + process supervision, REINFORCE) on ONE GPU: Qwen2.5-Math-1.5B-Instruct.
# Same algorithm flags as scripts/treerl-qw14b.sh; actor (DeepSpeed ZeRO-2, CPU Adam), and the vLLM engine
# share the GPU. See mid_submission/RL_CHANGES.md.
#   GPU=0 STEPS=100 nohup scripts/treerl-qwen1.5b-1gpu.sh > /tmp/treerl_rl.log 2>&1 &
# Baselines (same code; only the tree shape differs, TREE="M N L T"):
#   TreeRL (EPTree 6,2,1,2; 16 of its 30 leaves trained on):  TREE="6 2 1 2" NUM_TRACE=16
#   ChainRL (16 i.i.d. chains, all trained on):              TREE="16 0 0 0" NUM_TRACE=16 TAG=...-chainrl
#   GRPO (same 16 chains, group-normalized advantage):       TREE="16 0 0 0" NUM_TRACE=16 ADV=grpo TAG=...-grpo
# Reduced budget (see RL_CHANGES.md): ROLLOUT=8 NUM_TRACE=8 TREE="6 2 1 2" (or "8 0 0 0") MAX_LEN=2048 STEPS=40
#   DATA=mid_submission/data/train_30k_mixed.jsonl  (from mid_submission/solve_rate.py --keep_mixed)
set -x
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd $ROOT
source ~/likhit/.venv/bin/activate

GPU=${GPU:-0}
STEPS=${STEPS:-100}          # training steps; each step = ROLLOUT prompts x NUM_TRACE leaves
ROLLOUT=${ROLLOUT:-16}       # prompts per step (paper: 16)
NUM_TRACE=${NUM_TRACE:-16}   # leaves per tree used for training (paper: 16 of the 30 EPTree leaves)
TREE=(${TREE:-6 2 1 2})      # EPTree M N L T; "16 0 0 0" = ChainRL (M i.i.d. chains; needs M >= NUM_TRACE)
ADV=${ADV:-treerl}            # advantage: treerl (tree values) | grpo (needs TREE="G 0 0 0"; mid_submission/GRPO_CHANGES.md)
MAX_LEN=${MAX_LEN:-3072}     # max response tokens (Task 2: median 682, 1.1% over 2048)
RESUME=${RESUME:-}             # non-empty: resume from the newest $SAVE_DIR/_actor_global_step* (must exist)
DATA=$(realpath "${DATA:-$ROOT/datasets/train/train_30k.jsonl}")  # absolute: Ray workers have their own cwd
TAG=${TAG:-qwen1.5b-math-treerl-6-2-1-2}
SAVE_DIR=${SAVE_DIR:-$ROOT/ckpt/$TAG}
mkdir -p $SAVE_DIR

export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 RAY_DEDUP_LOGS=0

python train_reinforce_ray.py \
    --actor_num_nodes 1 --actor_num_gpus_per_node 1 \
    --ref_num_nodes 1 --ref_num_gpus_per_node 1 \
    --reward_num_nodes 0 \
    --vllm_num_engines 1 --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization ${VLLM_MEM:-0.35} --enable_prefix_caching \
    --pretrain ${MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct} \
    --reward_pretrain ${MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct} \
    --save_path $SAVE_DIR --ckpt_path $SAVE_DIR --save_steps ${SAVE_STEPS:-20} \
    --micro_train_batch_size 1 \
    --train_batch_size $((ROLLOUT * NUM_TRACE)) \
    --micro_rollout_batch_size $ROLLOUT \
    --rollout_batch_size $ROLLOUT \
    --inference_batch_size ${INFER_BS:-2} \
    --num_episodes 1 \
    --max_samples $((STEPS * ROLLOUT)) \
    --prompt_max_len 1024 \
    --generate_max_len $MAX_LEN \
    --zero_stage 2 --adam_offload --bf16 --gradient_checkpointing \
    --actor_learning_rate ${LR:-1.5e-6} --lr_scheduler_type cosine --min_actor_learning_rate_lr 1 --l2 0.1 \
    --init_kl_coef 0 \
    --prompt_data $DATA,1 \
    --input_key text --label_key label --source_key data_type \
    --top_p 0.95 --temperature 1.0 \
    --num_trace_per_sample $NUM_TRACE \
    --task_type qwen-math-reinforce \
    --remote_rm_url $ROOT/scripts/remote_reward_url.json \
    --advantage_estimator $ADV \
    --use_mcts --use_entropy_tree --m ${TREE[0]} --n ${TREE[1]} --l ${TREE[2]} --t ${TREE[3]} \
    --process_supervision --use_state_value_reward --use_pure_binary \
    --use_weighted_value --weighted_value_style sqrt \
    --mask_repeated_samples \
    --correct_bonus_ratio 1 --correct_bonus_threshold 0 \
    --perf ${RESUME:+--resume} \
    --wandb_run_name $TAG
