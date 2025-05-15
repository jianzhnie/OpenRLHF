#!/bin/bash

set -x

date=`date '+%Y%m%d_%H%M%S'`

export HCCL_CONNECT_TIMEOUT=1200
work_dir="work_dir/drgrpo/${date}"
# 如果work_dir不存，则创建该目录
if [ ! -d "./${work_dir}" ]; then
    mkdir -p ./${work_dir}
fi

read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain /root/llmtuner/hfhub/models/Qwen/Qwen2.5-Math-7B \
   --reward_func_names accuracy \
   --save_path ./${work_dir}/Qwen2.5-Math-7B \
   --ckpt_path ./${work_dir}/Qwen2.5-Math-7B/ckpt \
   --save_hf_ckpt \
   --max_ckpt_num 1000 \
   --save_steps 5 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --num_episodes 3 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 2048 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --value_clip 0.2 \
   --gamma 1.0 \
   --use_kl_loss \
   --init_kl_coef 0.001 \
   --kl_estimator k3 \
   --kl_target 6.0 \
   --kl_horizon 10000 \
   --advantage_estimator group_norm \
   --system_prompt qwen_math \
   --policy_loss_fn grpo \
   --correct_reward 1.0 \
   --incorrect_reward 0.0 \
   --cosine_min_value_wrong -0.5   \
   --cosine_max_value_wrong 0.0   \
   --cosine_min_value_correct 0.6   \
   --cosine_max_value_correct 1.0   \
   --cosine_max_len 2048 \
   --prompt_data /root/llmtuner/hfhub/datasets/math_lvl3to5_8k/train.jsonl \
   --dataset_config default \
   --input_key problem \
   --label_key solution \
   --apply_chat_template \
   --max_samples 100000 \
   --normalize_reward \
   --adam_offload \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_tensorboard ./${work_dir}/Qwen2.5-Math-7B/tb_log \
   --use_wandb 8ad58a961091cd50e8ed0b00d9060e502209a2b5 \
   --wandb_project pcl-grpo \
   --wandb_group grpo-baseline \
   --wandb_run_name Qwen2.5-Math-7B-grpo-baseline
EOF

    # --packing_samples
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --remote_rm_url http://localhost:5000/get_reward

if [[ ${1} != "slurm" ]]; then
    nohup deepspeed --module $training_commands > ${work_dir}/train.log 2>&1 &
fi

