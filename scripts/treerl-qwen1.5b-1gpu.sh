#!/bin/bash
# TreeRL (EPTree + process supervision, REINFORCE) on ONE GPU: Qwen2.5-Math-1.5B-Instruct.
# Same algorithm flags as scripts/treerl-qw14b.sh; actor (DeepSpeed ZeRO-2, CPU Adam), and the vLLM engine
# share the GPU. See mid_submission/RL_CHANGES.md.
#   GPU=0 STEPS=100 nohup scripts/treerl-qwen1.5b-1gpu.sh > /tmp/treerl_rl.log 2>&1 &
set -x
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd $ROOT
source ~/likhit/.venv/bin/activate

GPU=${GPU:-0}
STEPS=${STEPS:-100}          # training steps; each step = ROLLOUT prompts x NUM_TRACE leaves
ROLLOUT=${ROLLOUT:-16}       # prompts per step (paper: 16)
NUM_TRACE=${NUM_TRACE:-16}   # leaves per tree used for training (paper: 16 of the 30 EPTree leaves)
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
    --generate_max_len 3072 \
    --zero_stage 2 --adam_offload --bf16 --gradient_checkpointing \
    --actor_learning_rate 1.5e-6 --lr_scheduler_type cosine --min_actor_learning_rate_lr 1 --l2 0.1 \
    --init_kl_coef 0 \
    --prompt_data $ROOT/datasets/train/train_30k.jsonl,1 \
    --input_key text --label_key label --source_key data_type \
    --top_p 0.95 --temperature 1.0 \
    --num_trace_per_sample $NUM_TRACE \
    --task_type qwen-math-reinforce \
    --remote_rm_url $ROOT/scripts/remote_reward_url.json \
    --use_mcts --use_entropy_tree --m 6 --n 2 --l 1 --t 2 \
    --process_supervision --use_state_value_reward --use_pure_binary \
    --use_weighted_value --weighted_value_style sqrt \
    --mask_repeated_samples \
    --correct_bonus_ratio 1 --correct_bonus_threshold 0 \
    --perf \
    --wandb_run_name $TAG
