#!/bin/bash
set -x

# 从外部传入的参数
NNODES=$1
NODE_RANK=$2
NPUS_PER_NODE=$3
MASTER_ADDR=$4
MASTER_PORT=$5


DISTRIBUTED_ARGS="
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --nproc_per_node $NPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT  
"

# 设置环境变量
work_dir="./work_dir/drgrpo_math_lvl3to5/"
# 如果work_dir不存，则创建该目录
if [ ! -d "${work_dir}" ]; then
    mkdir -p ${work_dir}
fi

# 激活conda环境
source /root/llmtuner/miniconda3/bin/activate
conda activate rlhf

# 定义训练命令
torchrun \
  $DISTRIBUTED_ARGS \
   openrlhf/cli/train_ppo.py \
   --pretrain /root/llmtuner/hfhub/models/Qwen/Qwen2.5-32B \
   --reward_func_names accuracy \
   --save_path ${work_dir}/Qwen2.5-32B \
   --ckpt_path ${work_dir}/Qwen2.5-32B/ckpt \
   --save_hf_ckpt \
   --max_ckpt_num 1000 \
   --save_steps 10 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 1 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --num_episodes 5 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 2048 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --value_clip 0.2 \
   --gamma 1.0 \
   --init_kl_coef 0 \
   --advantage_estimator reinforce \
   --system_prompt qwen_math \
   --policy_loss_fn drgrpo \
   --eps_clip_high 0.28 \
   --correct_reward 1.0 \
   --incorrect_reward 0.0 \
   --prompt_data /root/llmtuner/hfhub/datasets/math_lvl3to5_8k/train.jsonl \
   --input_key problem \
   --label_key solution \
   --dataset_config default \
   --apply_chat_template \
   --max_samples 1000000 \
   --normalize_reward \
   --adam_offload \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_tensorboard  ${work_dir}/Qwen2.5-32B/tb_log \
   --use_wandb 8ad58a961091cd50e8ed0b00d9060e502209a2b5 \
   --wandb_project pcl-grpo \
   --wandb_group drgrpo \
   --wandb_run_name Qwen2.5-32B-drgrpo > ${work_dir}/Qwen2.5-32B/train_node_${NODE_RANK}.log 2>&1