#!/bin/bash

set -x

date=`date '+%Y%m%d_%H%M%S'`

export HCCL_CONNECT_TIMEOUT=1200
work_dir="/root/llm_workspace/models/grpo_naohai-2B_gsm8k/${date}"
# 如果work_dir不存，则创建该目录
if [ ! -d "${work_dir}" ]; then
    mkdir -p ${work_dir}
fi

read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain /root/llmtuner/hfhub/PengchengMind/naohai-2B-v7-unformat-sft_exp02 \
   --reward_func_names accuracy \
   --save_path ${work_dir}/naohai-2B \
   --ckpt_path ${work_dir}/naohai-2B/ckpt \
   --save_hf_ckpt \
   --max_ckpt_num 100 \
   --save_steps 20 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 4 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --num_episodes 3 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --gamma 1.0 \
   --init_kl_coef 0 \
   --advantage_estimator group_norm \
   --system_prompt qwen_math \
   --policy_loss_fn grpo \
   --correct_reward 1.0 \
   --incorrect_reward 0.0 \
   --cosine_min_value_wrong -0.5   \
   --cosine_max_value_wrong 0.0   \
   --cosine_min_value_correct 0.6   \
   --cosine_max_value_correct 1.0   \
   --cosine_max_len 1024 \
   --prompt_data /root/llmtuner/hfhub/datasets/openai/gsm8k \
   --dataset_config main \
   --input_key question \
   --label_key answer \
   --max_samples 100000 \
   --normalize_reward \
   --adam_offload \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_tensorboard ${work_dir}/naohai-2B/tb_log \
   --use_wandb 8ad58a961091cd50e8ed0b00d9060e502209a2b5 \
   --wandb_project pcl-grpo-gsm8k-naohai-2B \
   --wandb_group grpo \
   --wandb_run_name naohai-2B-grpo
EOF

    # --packing_samples
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --remote_rm_url http://localhost:5000/get_reward

if [[ ${1} != "slurm" ]]; then
    nohup deepspeed --module $training_commands > ${work_dir}/train.log 2>&1 &
fi